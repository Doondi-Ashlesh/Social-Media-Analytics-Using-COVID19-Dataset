"""
COVID-19 Twitter Discourse Analysis Pipeline - Enhanced Version
Minimal documentation, production-ready implementation
"""

import os
import sys
import json
import hashlib
import logging
import warnings
import platform
import psutil
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import re
import nltk
from langdetect import detect, LangDetectException
from fuzzywuzzy import fuzz

from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import umap
import hdbscan
from bertopic import BERTopic

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)


class CovidTwitterAnalysisPipeline:
    
    COLUMN_MAPPING = {
        'text': ['text', 'tweet', 'content', 'original_text', 'full_text'],
        'created_at': ['created_at', 'timestamp', 'date', 'created'],
        'author': ['user', 'username', 'author', 'original_author', 'screen_name'],
        'id': ['id', 'tweet_id', 'status_id'],
        'retweet_count': ['retweet_count', 'retweets', 'rt_count'],
        'favorite_count': ['favorite_count', 'favorites', 'likes', 'like_count'],
        'lang': ['lang', 'language']
    }
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.data = {}
        self.embeddings = None
        self.models = {}
        self.results = {}
        
        self._create_output_dirs()
    
    def _get_default_config(self) -> Dict:
        return {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dim': 384,
            'batch_size': 32,
            'chunk_size': 10000,
            'hdbscan': {
                'min_cluster_size': 50,
                'min_samples': 15,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'prediction_data': True
            },
            'umap': {
                'n_components': 5,
                'n_neighbors': 15,
                'min_dist': 0.0,
                'metric': 'cosine',
                'random_state': RANDOM_SEED
            },
            'isolation_forest': {
                'n_estimators': 100,
                'max_samples': 256,
                'contamination': 'auto',
                'random_state': RANDOM_SEED
            },
            'bot_threshold': 0.75,
            'similarity_threshold': 0.85,
            'significance_level': 0.05,
            'output_dir': 'outputs',
            'use_checkpoints': True
        }
    
    def _create_output_dirs(self):
        dirs = [
            'outputs/data', 'outputs/data/intermediate', 'outputs/reports',
            'outputs/visualizations', 'outputs/reproducibility', 'outputs/validation'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: DATA PREPROCESSING
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_and_preprocess_data(self, data_paths: List[str]) -> pd.DataFrame:
        logger.info("="*80)
        logger.info("PHASE 1: DATA PREPROCESSING AND PREPARATION")
        logger.info("="*80)
        
        dfs = []
        for path in data_paths:
            try:
                df = pd.read_csv(path, encoding='utf-8', low_memory=False)
                df = self._standardize_columns(df)
                phase = self._extract_phase_from_path(path)
                df['phase'] = phase
                dfs.append(df)
                logger.info(f"Loaded {len(df):,} tweets from {path}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No datasets were successfully loaded")
        
        df_combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total tweets loaded: {len(df_combined):,}")
        
        df_dedup = self._deduplicate_tweets(df_combined)
        df_clean = self._normalize_text(df_dedup)
        df_english = self._filter_language(df_clean)
        df_processed = self._extract_metadata(df_english)
        
        self.data['processed'] = df_processed
        df_processed.to_csv('outputs/data/processed_tweets.csv', index=False)
        df_processed.to_parquet('outputs/data/processed_tweets.parquet', index=False)
        
        logger.info(f"Preprocessing complete. Final dataset size: {len(df_processed):,}")
        return df_processed
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for standard_name, possible_names in self.COLUMN_MAPPING.items():
            for col in df.columns:
                if col.lower() in [p.lower() for p in possible_names]:
                    df.rename(columns={col: standard_name}, inplace=True)
                    break
        
        if 'original_text' not in df.columns and 'text' in df.columns:
            df['original_text'] = df['text']
        if 'original_author' not in df.columns and 'author' in df.columns:
            df['original_author'] = df['author']
        
        return df
    
    def _extract_phase_from_path(self, path: str) -> str:
        if 'Apr-Jun 2020' in path or 'Phase1' in path or 'phase1' in path:
            return 'Phase1'
        elif 'Aug-Oct 2020' in path or 'Phase2' in path or 'phase2' in path:
            return 'Phase2'
        elif 'Apr-Jun 2021' in path or 'Phase3' in path or 'phase3' in path:
            return 'Phase3'
        return 'Unknown'
    
    def _deduplicate_tweets(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing duplicates...")
        initial_size = len(df)
        
        df['text_hash'] = df['original_text'].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest() if pd.notna(x) else None
        )
        df_dedup = df.drop_duplicates(subset=['text_hash'], keep='first')
        
        if len(df_dedup) > 10000:
            try:
                from datasketch import MinHash, MinHashLSH
                
                def get_minhash(text, num_perm=128):
                    m = MinHash(num_perm=num_perm)
                    for word in str(text).split():
                        m.update(word.encode('utf-8'))
                    return m
                
                lsh = MinHashLSH(threshold=0.9, num_perm=128)
                minhashes = {}
                
                sample_size = min(5000, len(df_dedup))
                sample_indices = np.random.choice(df_dedup.index, sample_size, replace=False)
                
                for idx in tqdm(sample_indices, desc="MinHash indexing"):
                    text = df_dedup.loc[idx, 'cleaned_text'] if 'cleaned_text' in df_dedup.columns else df_dedup.loc[idx, 'original_text']
                    if pd.notna(text):
                        m = get_minhash(text)
                        lsh.insert(idx, m)
                        minhashes[idx] = m
                
                duplicates = set()
                for idx, m in tqdm(minhashes.items(), desc="Finding near-duplicates"):
                    if idx in duplicates:
                        continue
                    similar = lsh.query(m)
                    if len(similar) > 1:
                        duplicates.update(similar[1:])
                
                df_dedup = df_dedup[~df_dedup.index.isin(duplicates)]
            except ImportError:
                logger.warning("datasketch not available, using basic deduplication")
        
        logger.info(f"Removed {initial_size - len(df_dedup):,} duplicates")
        return df_dedup
    
    def _normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Normalizing text...")
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = text.lower()
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\.[a-zA-Z0-9\.\-]+\.[a-zA-Z]+', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\bcovid\b|\bcovid[\-\s]?19\b|\bcoronavirus\b', 'covid19', text, flags=re.IGNORECASE)
            text = re.sub(r'^rt\s+@\w+:\s*', '', text)
            return text.strip()
        
        df['cleaned_text'] = df['original_text'].apply(clean_text)
        df = df[df['cleaned_text'].str.len() > 0]
        
        return df
    
    def _filter_language(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering for English tweets...")
        
        def detect_language(text):
            try:
                if pd.notna(text) and len(text) > 10:
                    return detect(text)
                return 'unknown'
            except LangDetectException:
                return 'unknown'
        
        if 'lang' in df.columns:
            df_english = df[df['lang'] == 'en'].copy()
        else:
            df['detected_lang'] = df['cleaned_text'].apply(detect_language)
            df_english = df[df['detected_lang'] == 'en'].copy()
        
        logger.info(f"Retained {len(df_english):,} English tweets")
        return df_english
    
    def _extract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting metadata...")
        
        df['timestamp'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        df['retweet_count'] = pd.to_numeric(df['retweet_count'], errors='coerce').fillna(0)
        df['favorite_count'] = pd.to_numeric(df['favorite_count'], errors='coerce').fillna(0)
        df['engagement_score'] = df['retweet_count'] + df['favorite_count']
        
        df['hashtag_list'] = df['original_text'].apply(
            lambda x: re.findall(r'#\w+', str(x).lower()) if pd.notna(x) else []
        )
        df['hashtag_count'] = df['hashtag_list'].apply(len)
        
        df['mention_list'] = df['original_text'].apply(
            lambda x: re.findall(r'@\w+', str(x).lower()) if pd.notna(x) else []
        )
        df['mention_count'] = df['mention_list'].apply(len)
        
        df['text_length'] = df['cleaned_text'].str.len()
        
        return df
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: SEMANTIC EMBEDDING GENERATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        logger.info("="*80)
        logger.info("PHASE 2: SEMANTIC EMBEDDING GENERATION")
        logger.info("="*80)
        
        embedding_path = 'outputs/data/intermediate/embeddings.npy'
        if os.path.exists(embedding_path):
            logger.info("Loading existing embeddings...")
            self.embeddings = np.load(embedding_path)
            return self.embeddings
        
        logger.info(f"Loading embedding model: {self.config['embedding_model']}")
        model = SentenceTransformer(self.config['embedding_model'], device=self.device)
        self.models['embedding'] = model
        
        texts = df['cleaned_text'].tolist()
        total_texts = len(texts)
        chunk_size = self.config['chunk_size']
        
        all_embeddings = []
        
        for chunk_start in range(0, total_texts, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_texts)
            chunk_texts = texts[chunk_start:chunk_end]
            
            logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_texts-1)//chunk_size + 1}")
            
            batch_size = self.config['batch_size']
            chunk_embeddings = []
            
            for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Generating embeddings"):
                batch = chunk_texts[i:i+batch_size]
                try:
                    batch_embeddings = model.encode(
                        batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        batch_size=batch_size
                    )
                    chunk_embeddings.append(batch_embeddings.cpu().numpy())
                    
                    if torch.cuda.is_available() and i % (batch_size * 10) == 0:
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Error in batch {i}: {e}, using zero embeddings")
                    chunk_embeddings.append(np.zeros((len(batch), self.config['embedding_dim'])))
            
            chunk_matrix = np.vstack(chunk_embeddings)
            all_embeddings.append(chunk_matrix)
        
        embedding_matrix = np.vstack(all_embeddings)
        self.embeddings = embedding_matrix
        
        np.save(embedding_path, embedding_matrix)
        
        logger.info(f"Generated embeddings with shape: {embedding_matrix.shape}")
        return embedding_matrix
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: TOPIC DISCOVERY
    # ═══════════════════════════════════════════════════════════════════════
    
    def discover_topics(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
        logger.info("="*80)
        logger.info("PHASE 3: UNSUPERVISED TOPIC DISCOVERY AND MODELING")
        logger.info("="*80)
        
        umap_model = umap.UMAP(
            n_components=self.config['umap']['n_components'],
            n_neighbors=self.config['umap']['n_neighbors'],
            min_dist=self.config['umap']['min_dist'],
            metric=self.config['umap']['metric'],
            random_state=self.config['umap']['random_state']
        )
        
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.config['hdbscan']['min_cluster_size'],
            min_samples=self.config['hdbscan']['min_samples'],
            metric=self.config['hdbscan']['metric'],
            cluster_selection_method=self.config['hdbscan']['cluster_selection_method'],
            prediction_data=self.config['hdbscan']['prediction_data']
        )
        
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics='auto',
            calculate_probabilities=True,
            verbose=True
        )
        
        logger.info("Fitting BERTopic model...")
        topics, probs = topic_model.fit_transform(df['cleaned_text'].tolist(), embeddings)
        
        df['topic'] = topics
        df['topic_probability'] = probs.max(axis=1) if len(probs.shape) > 1 else probs
        
        topic_info = topic_model.get_topic_info()
        
        topic_results = {
            'model': topic_model,
            'topic_info': topic_info,
            'topics': topics,
            'probabilities': probs,
            'num_topics': len(set(topics)) - (1 if -1 in topics else 0)
        }
        
        topic_model.save("outputs/data/intermediate/topic_model")
        topic_results['characterization'] = self._characterize_topics(df, topic_model, embeddings)
        
        self.results['topics'] = topic_results
        
        logger.info(f"Discovered {topic_results['num_topics']} topics")
        return topic_results
    
    def _characterize_topics(self, df: pd.DataFrame, topic_model: BERTopic, embeddings: np.ndarray) -> Dict:
        logger.info("Characterizing topics...")
        
        characterizations = {}
        topics = df['topic'].unique()
        topics = [t for t in topics if t != -1]
        
        for topic_id in tqdm(topics, desc="Characterizing topics"):
            keywords = topic_model.get_topic(topic_id)[:10] if topic_id != -1 else []
            
            topic_tweets = df[df['topic'] == topic_id]
            if len(topic_tweets) > 0:
                topic_embeddings = embeddings[df['topic'] == topic_id]
                centroid = topic_embeddings.mean(axis=0)
                
                distances = np.linalg.norm(topic_embeddings - centroid, axis=1)
                closest_indices = np.argsort(distances)[:4]
                representative_tweets = topic_tweets.iloc[closest_indices]['cleaned_text'].tolist()
            else:
                representative_tweets = []
            
            characterizations[topic_id] = {
                'keywords': keywords,
                'representative_tweets': representative_tweets,
                'size': len(topic_tweets),
                'percentage': len(topic_tweets) / len(df) * 100
            }
        
        return characterizations
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: AFFECTIVE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    def analyze_sentiment_emotion(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("="*80)
        logger.info("PHASE 4: AUTOMATED AFFECTIVE ANALYSIS")
        logger.info("="*80)

        df.reset_index(drop=True, inplace=True)
        
        logger.info("Performing sentiment analysis...")
        sentiment_results = self._analyze_sentiment(df)
        
        logger.info("Performing emotion analysis...")
        emotion_results = self._analyze_emotion(df)
        
        df = pd.concat([df, sentiment_results, emotion_results], axis=1)
        affective_summary = self._aggregate_affective_analysis(df)
        self.results['affective'] = affective_summary
        
        return df
    
    def _analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        try:
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_length=512,
                truncation=True
            )
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}. Using fallback.")
            return self._fallback_sentiment_analysis(df)
        
        batch_size = 16
        texts = df['cleaned_text'].tolist()
        
        all_results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment analysis"):
            batch = texts[i:i+batch_size]
            batch_truncated = [text[:512] if text else "" for text in batch]
            
            try:
                results = sentiment_pipeline(batch_truncated)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Error in batch {i//batch_size}: {e}")
                neutral_results = [{'label': 'neutral', 'score': 0.5}] * len(batch)
                all_results.extend(neutral_results)
        
        sentiment_df = pd.DataFrame({
            'sentiment_label': [r['label'] for r in all_results],
            'sentiment_score': [r['score'] for r in all_results]
        }, index=df.index)
        
        label_map = {
            'negative': -1, 'neutral': 0, 'positive': 1,
            'LABEL_0': -1, 'LABEL_1': 0, 'LABEL_2': 1,
            'neg': -1, 'neu': 0, 'pos': 1
        }
        
        sentiment_df['sentiment_numeric'] = sentiment_df['sentiment_label'].apply(
            lambda x: label_map.get(x.lower(), 0)
        )

        return sentiment_df
    
    def _fallback_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        try:
            sia = SentimentIntensityAnalyzer()
            
            sentiments = []
            for text in tqdm(df['cleaned_text'], desc="Fallback sentiment"):
                scores = sia.polarity_scores(text)
                compound = scores['compound']
                
                if compound >= 0.05:
                    label, numeric = 'positive', 1
                elif compound <= -0.05:
                    label, numeric = 'negative', -1
                else:
                    label, numeric = 'neutral', 0
                
                sentiments.append({
                    'sentiment_label': label,
                    'sentiment_score': abs(compound),
                    'sentiment_numeric': numeric
                })
            
            return pd.DataFrame(sentiments)
        
        except Exception as e:
            logger.error(f"Fallback sentiment analysis failed: {e}")
            return pd.DataFrame({
                'sentiment_label': ['neutral'] * len(df),
                'sentiment_score': [0.5] * len(df),
                'sentiment_numeric': [0] * len(df)
            })
    
    def _analyze_emotion(self, df: pd.DataFrame) -> pd.DataFrame:
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        
        try:
            emotion_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                top_k=None
            )
        except Exception as e:
            logger.warning(f"Failed to load emotion model: {e}")
            return self._fallback_emotion_analysis(df)
        
        batch_size = 16
        texts = df['cleaned_text'].tolist()
        
        all_results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Emotion analysis"):
            batch = texts[i:i+batch_size]
            batch_truncated = [text[:512] for text in batch]
            
            try:
                results = emotion_pipeline(batch_truncated)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Error in emotion batch: {e}")
                all_results.extend([[{'label': 'neutral', 'score': 1.0}]] * len(batch))
        
        emotion_df = pd.DataFrame({
            'emotion_label': [max(r, key=lambda x: x['score'])['label'] for r in all_results],
            'emotion_score': [max(r, key=lambda x: x['score'])['score'] for r in all_results]
        }, index=df.index)
        
        emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        for emotion in emotion_labels:
            emotion_df[f'emotion_{emotion}'] = [
                next((e['score'] for e in r if e['label'] == emotion), 0.0)
                for r in all_results
            ]

        return emotion_df
    
    def _fallback_emotion_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        emotion_df = pd.DataFrame({
            'emotion_label': ['neutral'] * n,
            'emotion_score': [1.0] * n,
        })
        
        # Add all emotion probabilities
        emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        for emotion in emotions:
            emotion_df[f'emotion_{emotion}'] = [1.0 if emotion == 'neutral' else 0.0] * n

        emotion_df.reset_index(drop=True, inplace=True)
        return emotion_df

    
    def _aggregate_affective_analysis(self, df: pd.DataFrame) -> Dict:
        affective_summary = {}
        
        affective_summary['overall_sentiment'] = {
            'positive': (df['sentiment_numeric'] == 1).mean(),
            'neutral': (df['sentiment_numeric'] == 0).mean(),
            'negative': (df['sentiment_numeric'] == -1).mean(),
            'avg_score': df['sentiment_numeric'].mean()
        }
        
        emotion_dist = df['emotion_label'].value_counts(normalize=True).to_dict()
        affective_summary['overall_emotion'] = emotion_dist
        
        topic_affective = {}
        for topic_id in df['topic'].unique():
            if topic_id == -1:
                continue
                
            topic_df = df[df['topic'] == topic_id]
            topic_affective[topic_id] = {
                'sentiment': {
                    'positive': (topic_df['sentiment_numeric'] == 1).mean(),
                    'neutral': (topic_df['sentiment_numeric'] == 0).mean(),
                    'negative': (topic_df['sentiment_numeric'] == -1).mean(),
                    'avg_score': topic_df['sentiment_numeric'].mean()
                },
                'emotion': topic_df['emotion_label'].value_counts(normalize=True).to_dict()
            }
        
        affective_summary['topic_affective'] = topic_affective
        
        return affective_summary
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: ANOMALY DETECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def detect_anomalies(self, df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        logger.info("="*80)
        logger.info("PHASE 5: ANOMALY DETECTION AND CHARACTERIZATION")
        logger.info("="*80)
        
        optimal_params = self._optimize_isolation_forest(embeddings)
        
        logger.info("Detecting anomalies...")
        iforest = IsolationForest(**optimal_params)
        iforest.fit(embeddings)
        
        anomaly_scores = iforest.score_samples(embeddings)
        anomaly_labels = iforest.predict(embeddings)
        
        df['anomaly_score'] = anomaly_scores
        df['anomaly_label'] = (anomaly_labels == -1).astype(int)
        
        anomaly_char = self._characterize_anomalies(df, embeddings)
        
        self.results['anomalies'] = {
            'total_anomalies': df['anomaly_label'].sum(),
            'anomaly_rate': df['anomaly_label'].mean(),
            'characterization': anomaly_char
        }
        
        logger.info(f"Detected {df['anomaly_label'].sum():,} anomalies ({df['anomaly_label'].mean()*100:.2f}%)")
        
        return df
    
    def _optimize_isolation_forest(self, embeddings: np.ndarray) -> Dict:
        logger.info("Optimizing Isolation Forest parameters...")
        
        best_params = self.config['isolation_forest'].copy()
        best_stability = 0
        
        for n_trees in [100, 150, 200]:
            stability_scores = []
            
            for run in range(3):
                iforest = IsolationForest(
                    n_estimators=n_trees,
                    max_samples=256,
                    contamination='auto',
                    random_state=RANDOM_SEED + run
                )
                sample_size = min(10000, len(embeddings))
                iforest.fit(embeddings[:sample_size])
                scores = iforest.score_samples(embeddings[:sample_size])
                stability_scores.append(scores)
            
            score_variance = np.var(stability_scores, axis=0).mean()
            stability = 1 / (1 + score_variance)
            
            if stability > best_stability:
                best_stability = stability
                best_params['n_estimators'] = n_trees
        
        logger.info(f"Optimal n_estimators: {best_params['n_estimators']}")
        return best_params
    
    def _characterize_anomalies(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
        logger.info("Characterizing anomalies...")
        
        anomalous = df[df['anomaly_label'] == 1]
        normal = df[df['anomaly_label'] == 0]
        
        characterization = {}
        
        anomaly_prone_topics = []
        for topic in df['topic'].unique():
            if topic == -1:
                continue
            topic_anomaly_rate = (df[df['topic'] == topic]['anomaly_label'].mean())
            overall_rate = df['anomaly_label'].mean()
            if topic_anomaly_rate > 2 * overall_rate:
                anomaly_prone_topics.append(topic)
        
        characterization['anomaly_prone_topics'] = anomaly_prone_topics
        
        characterization['linguistic'] = {
            'avg_length_anomalous': anomalous['text_length'].mean() if len(anomalous) > 0 else 0,
            'avg_length_normal': normal['text_length'].mean() if len(normal) > 0 else 0,
            'hashtag_count_anomalous': anomalous['hashtag_count'].mean() if len(anomalous) > 0 else 0,
            'hashtag_count_normal': normal['hashtag_count'].mean() if len(normal) > 0 else 0,
            'mention_count_anomalous': anomalous['mention_count'].mean() if len(anomalous) > 0 else 0,
            'mention_count_normal': normal['mention_count'].mean() if len(normal) > 0 else 0
        }
        
        if 'timestamp' in df.columns and len(anomalous) > 0:
            anomaly_temporal = anomalous.groupby(pd.Grouper(key='timestamp', freq='D')).size()
            if len(anomaly_temporal) > 0:
                bursts = anomaly_temporal[anomaly_temporal > anomaly_temporal.quantile(0.95)]
                characterization['temporal_bursts'] = bursts.index.tolist()
        
        return characterization
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 6: BOT DETECTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def detect_bots(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("="*80)
        logger.info("PHASE 6: BOT DETECTION AND IMPACT ANALYSIS")
        logger.info("="*80)
        
        logger.info("Detecting potential bots using heuristics...")
        
        user_stats = df.groupby('original_author').agg({
            'id': 'count',
            'cleaned_text': lambda x: len(set(x)) / len(x) if len(x) > 0 else 1,
            'retweet_count': 'mean',
            'favorite_count': 'mean',
            'hashtag_count': 'mean',
            'mention_count': 'mean',
            'timestamp': lambda x: (x.max() - x.min()).total_seconds() / 3600 if len(x) > 1 else 0
        }).rename(columns={'id': 'tweet_count', 'cleaned_text': 'unique_ratio'})
        
        bot_scores = pd.DataFrame(index=user_stats.index)
        bot_scores['volume_score'] = (user_stats['tweet_count'] > user_stats['tweet_count'].quantile(0.95)).astype(float)
        bot_scores['repetitive_score'] = (user_stats['unique_ratio'] < 0.3).astype(float)
        bot_scores['hashtag_score'] = (user_stats['hashtag_count'] > user_stats['hashtag_count'].quantile(0.9)).astype(float)
        bot_scores['bot_score'] = bot_scores.mean(axis=1)
        bot_scores['bot_label'] = (bot_scores['bot_score'] >= self.config['bot_threshold']).astype(int)
        
        df = df.merge(
            bot_scores[['bot_score', 'bot_label']], 
            left_on='original_author', 
            right_index=True, 
            how='left'
        )
        
        df['bot_score'] = df['bot_score'].fillna(0.5)
        df['bot_label'] = df['bot_label'].fillna(0)
        df['human_weight'] = 1 - df['bot_score']
        
        bot_impact = self._analyze_bot_impact(df)
        
        self.results['bots'] = {
            'bot_accounts': bot_scores['bot_label'].sum(),
            'bot_tweets': df['bot_label'].sum(),
            'bot_tweet_percentage': df['bot_label'].mean() * 100,
            'impact': bot_impact
        }
        
        logger.info(f"Identified {bot_scores['bot_label'].sum():,} potential bot accounts")
        logger.info(f"Bot tweets: {df['bot_label'].sum():,} ({df['bot_label'].mean()*100:.2f}%)")
        
        return df
    
    def _analyze_bot_impact(self, df: pd.DataFrame) -> Dict:
        impact = {}
        
        human_sentiment = df[df['bot_label'] == 0]['sentiment_numeric'].mean()
        bot_sentiment = df[df['bot_label'] == 1]['sentiment_numeric'].mean()
        overall_sentiment = df['sentiment_numeric'].mean()
        
        impact['sentiment'] = {
            'human_only': human_sentiment,
            'bot_only': bot_sentiment,
            'overall': overall_sentiment,
            'bot_influence': overall_sentiment - human_sentiment
        }
        
        bot_topic_dist = df[df['bot_label'] == 1]['topic'].value_counts(normalize=True)
        human_topic_dist = df[df['bot_label'] == 0]['topic'].value_counts(normalize=True)
        
        amplified_topics = []
        for topic in bot_topic_dist.index:
            if topic == -1:
                continue
            bot_rate = bot_topic_dist.get(topic, 0)
            human_rate = human_topic_dist.get(topic, 0)
            if human_rate > 0 and bot_rate / human_rate > 1.5:
                amplified_topics.append(topic)
        
        impact['amplified_topics'] = amplified_topics
        
        bot_tweets = df[df['bot_label'] == 1]
        if len(bot_tweets) > 100:
            sample_size = min(1000, len(bot_tweets))
            sample = bot_tweets.sample(n=sample_size, random_state=RANDOM_SEED)
            
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sample['cleaned_text'])
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            coordinated_pairs = np.sum(similarity_matrix > self.config['similarity_threshold'])
            impact['coordinated_campaigns'] = coordinated_pairs // 2
        else:
            impact['coordinated_campaigns'] = 0
        
        return impact
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 7: STATISTICAL VALIDATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        logger.info("="*80)
        logger.info("PHASE 7: STATISTICAL VALIDATION AND SIGNIFICANCE TESTING")
        logger.info("="*80)
        
        statistical_results = {}
        
        trend_results = self._mann_kendall_analysis(df)
        statistical_results['trends'] = trend_results
        
        changepoint_results = self._detect_changepoints(df)
        statistical_results['changepoints'] = changepoint_results
        
        if 'topics' in self.results:
            coherence_results = self._validate_topic_coherence(df)
            statistical_results['coherence'] = coherence_results
        
        self.results['statistics'] = statistical_results
        
        return statistical_results
    
    def _mann_kendall_analysis(self, df: pd.DataFrame) -> Dict:
        logger.info("Performing Mann-Kendall trend analysis...")
        
        daily_stats = df.groupby(pd.Grouper(key='timestamp', freq='D')).agg({
            'sentiment_numeric': 'mean',
            'anomaly_label': 'mean',
            'bot_label': 'mean',
            'engagement_score': 'mean'
        })
        
        trend_results = {}
        for column in daily_stats.columns:
            series = daily_stats[column].dropna()
            if len(series) > 10:
                tau, p_value = stats.kendalltau(range(len(series)), series)
                
                trend_results[column] = {
                    'tau': float(tau),
                    'p_value': float(p_value),
                    'significant': p_value < self.config['significance_level'],
                    'direction': 'increasing' if tau > 0 else 'decreasing'
                }
        
        return trend_results
    
    def _detect_changepoints(self, df: pd.DataFrame) -> List[Dict]:
        logger.info("Detecting changepoints...")
        
        daily_sentiment = df.groupby(pd.Grouper(key='timestamp', freq='D'))['sentiment_numeric'].mean()
        
        changepoints = []
        window_size = 7
        threshold = 2
        
        for i in range(window_size, len(daily_sentiment) - window_size):
            before = daily_sentiment[i-window_size:i]
            after = daily_sentiment[i:i+window_size]
            
            if len(before.dropna()) < 3 or len(after.dropna()) < 3:
                continue
            
            t_stat, p_value = stats.ttest_ind(before.dropna(), after.dropna())
            
            if p_value < self.config['significance_level'] and abs(t_stat) > threshold:
                changepoints.append({
                    'date': str(daily_sentiment.index[i]),
                    'metric': 'sentiment',
                    'before_mean': float(before.mean()),
                    'after_mean': float(after.mean()),
                    'magnitude': float(after.mean() - before.mean()),
                    'p_value': float(p_value)
                })
        
        return changepoints
    
    def _validate_topic_coherence(self, df: pd.DataFrame) -> Dict:
        logger.info("Validating topic coherence...")
        
        coherence_results = {}
        
        if 'characterization' in self.results.get('topics', {}):
            topics_char = self.results['topics']['characterization']
            
            coherence_scores = []
            for topic_id, info in topics_char.items():
                topic_tweets = df[df['topic'] == topic_id]['cleaned_text']
                if len(topic_tweets) > 0 and info['keywords']:
                    keyword_freq = sum(
                        sum(keyword[0].lower() in tweet.lower() 
                            for keyword in info['keywords'][:5])
                        for tweet in topic_tweets
                    ) / (len(topic_tweets) * 5)
                    coherence_scores.append(keyword_freq)
            
            if coherence_scores:
                coherence_results['avg_coherence'] = float(np.mean(coherence_scores))
                coherence_results['min_coherence'] = float(np.min(coherence_scores))
                coherence_results['max_coherence'] = float(np.max(coherence_scores))
            else:
                coherence_results = {'avg_coherence': 0, 'min_coherence': 0, 'max_coherence': 0}
        
        return coherence_results
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 8: VISUALIZATION AND OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    
    def generate_visualizations(self, df: pd.DataFrame) -> None:
        logger.info("="*80)
        logger.info("PHASE 8: INTEGRATED OUTPUT GENERATION AND VISUALIZATION")
        logger.info("="*80)
        
        self._create_temporal_plots(df)
        
        if 'topics' in self.results:
            self._create_topic_plots(df)
        
        if 'affective' in self.results:
            self._create_affective_plots(df)
        
        if 'anomalies' in self.results:
            self._create_anomaly_plots(df)
        
        self._create_interactive_timeline(df)
        self._create_topic_network(df)
        
        logger.info("Visualizations saved to outputs/visualizations/")
    
    def _create_temporal_plots(self, df: pd.DataFrame):
        logger.info("Creating temporal visualizations...")
        
        daily = df.groupby(pd.Grouper(key='timestamp', freq='D')).agg({
            'sentiment_numeric': 'mean',
            'anomaly_label': 'mean',
            'bot_label': 'mean',
            'id': 'count'
        }).rename(columns={'id': 'tweet_count'})
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        axes[0].plot(daily.index, daily['tweet_count'], color='steelblue', linewidth=2)
        axes[0].fill_between(daily.index, daily['tweet_count'], alpha=0.3, color='steelblue')
        axes[0].set_title('Daily Tweet Volume', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Tweets')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(daily.index, daily['sentiment_numeric'], color='green', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title('Sentiment Trend', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Average Sentiment')
        axes[1].set_ylim(-1, 1)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(daily.index, daily['anomaly_label'] * 100, color='red', linewidth=2)
        axes[2].set_title('Anomaly Rate', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Anomaly %')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(daily.index, daily['bot_label'] * 100, color='orange', linewidth=2)
        axes[3].set_title('Bot Activity', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Bot Tweet %')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_topic_plots(self, df: pd.DataFrame):
        logger.info("Creating topic visualizations...")
        
        topic_counts = df[df['topic'] != -1]['topic'].value_counts().head(20)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(topic_counts)), topic_counts.values, color='skyblue')
        
        if 'sentiment_numeric' in df.columns:
            for i, topic in enumerate(topic_counts.index):
                sentiment = df[df['topic'] == topic]['sentiment_numeric'].mean()
                color = 'green' if sentiment > 0.1 else 'red' if sentiment < -0.1 else 'gray'
                bars[i].set_color(color)
        
        ax.set_xlabel('Topic ID', fontsize=12)
        ax.set_ylabel('Number of Tweets', fontsize=12)
        ax.set_title('Top 20 Topics by Tweet Count', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(topic_counts)))
        ax.set_xticklabels(topic_counts.index, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/topic_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_affective_plots(self, df: pd.DataFrame):
        logger.info("Creating affective visualizations...")
        
        if 'topic' in df.columns and 'emotion_label' in df.columns:
            topics = df[df['topic'] != -1]['topic'].value_counts().head(15).index
            emotions = df['emotion_label'].unique()
            
            matrix = np.zeros((len(emotions), len(topics)))
            for i, emotion in enumerate(emotions):
                for j, topic in enumerate(topics):
                    count = len(df[(df['emotion_label'] == emotion) & (df['topic'] == topic)])
                    total = len(df[df['topic'] == topic])
                    matrix[i, j] = count / total if total > 0 else 0
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(matrix, 
                       xticklabels=topics,
                       yticklabels=emotions,
                       cmap='RdYlBu_r',
                       annot=True,
                       fmt='.2%',
                       cbar_kws={'label': 'Proportion'},
                       ax=ax)
            
            ax.set_title('Emotion Distribution Across Topics', fontsize=14, fontweight='bold')
            ax.set_xlabel('Topic ID', fontsize=12)
            ax.set_ylabel('Emotion', fontsize=12)
            
            plt.tight_layout()
            plt.savefig('outputs/visualizations/emotion_topic_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_anomaly_plots(self, df: pd.DataFrame):
        logger.info("Creating anomaly visualizations...")
        
        if self.embeddings is not None and len(self.embeddings) < 50000:
            sample_size = min(10000, len(df))
            sample_idx = np.random.choice(len(df), sample_size, replace=False)
            
            umap_2d = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
            embeddings_2d = umap_2d.fit_transform(self.embeddings[sample_idx])
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                      c=df.iloc[sample_idx]['anomaly_label'],
                                      cmap='coolwarm', alpha=0.6, s=10)
            axes[0].set_title('UMAP Projection - Anomalies', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('UMAP 1')
            axes[0].set_ylabel('UMAP 2')
            plt.colorbar(scatter1, ax=axes[0], label='Anomaly')
            
            if 'topic' in df.columns:
                scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                          c=df.iloc[sample_idx]['topic'],
                                          cmap='tab20', alpha=0.6, s=10)
                axes[1].set_title('UMAP Projection - Topics', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('UMAP 1')
                axes[1].set_ylabel('UMAP 2')
                plt.colorbar(scatter2, ax=axes[1], label='Topic')
            
            plt.tight_layout()
            plt.savefig('outputs/visualizations/umap_projections.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_interactive_timeline(self, df: pd.DataFrame):
        logger.info("Creating interactive timeline...")
        
        daily = df.groupby(pd.Grouper(key='timestamp', freq='D')).agg({
            'sentiment_numeric': 'mean',
            'anomaly_label': 'mean',
            'bot_label': 'mean',
            'id': 'count',
            'engagement_score': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Tweet Volume', 'Sentiment Trend', 'Anomaly Rate', 'Bot Activity'),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}]] * 4
        )
        
        fig.add_trace(
            go.Scatter(x=daily['timestamp'], y=daily['id'],
                      name='Tweet Count', fill='tozeroy',
                      line=dict(color='steelblue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily['timestamp'], y=daily['sentiment_numeric'],
                      name='Sentiment', line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily['timestamp'], y=daily['anomaly_label']*100,
                      name='Anomaly %', line=dict(color='red', width=2)),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=daily['timestamp'], y=daily['bot_label']*100,
                      name='Bot %', line=dict(color='orange', width=2)),
            row=4, col=1
        )
        
        fig.update_layout(
            height=1200,
            title_text="COVID-19 Twitter Discourse Timeline",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.write_html('outputs/visualizations/interactive_timeline.html')
    
    def _create_topic_network(self, df: pd.DataFrame):
        logger.info("Creating topic network...")
        
        df['week'] = df['timestamp'].dt.to_period('W')
        
        topics = df[df['topic'] != -1]['topic'].unique()
        if len(topics) == 0:
            return
        
        cooccurrence = np.zeros((len(topics), len(topics)))
        
        for week in df['week'].unique():
            week_topics = df[df['week'] == week]['topic'].value_counts()
            for i, t1 in enumerate(topics):
                for j, t2 in enumerate(topics):
                    if t1 in week_topics.index and t2 in week_topics.index:
                        cooccurrence[i, j] += 1
        
        G = nx.Graph()
        for i, topic in enumerate(topics):
            topic_size = len(df[df['topic'] == topic])
            G.add_node(topic, size=topic_size)
        
        threshold = np.percentile(cooccurrence, 75)
        for i in range(len(topics)):
            for j in range(i+1, len(topics)):
                if cooccurrence[i, j] > threshold:
                    G.add_edge(topics[i], topics[j], weight=cooccurrence[i, j])
        
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(
                go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                          mode='lines',
                          line=dict(width=0.5, color='#888'),
                          hoverinfo='none',
                          showlegend=False)
            )
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[f"Topic {node}" for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=[G.nodes[node]['size']/100 for node in G.nodes()],
                color=list(G.nodes()),
                colorscale='Viridis',
                line_width=2
            ),
            hovertemplate='<b>Topic %{text}</b><br>Size: %{marker.size}<extra></extra>'
        )
        
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title='Topic Co-occurrence Network',
                           showlegend=False,
                           hovermode='closest',
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        fig.write_html('outputs/visualizations/topic_network.html')
    
    def generate_reports(self, df: pd.DataFrame) -> None:
        logger.info("Generating analytical reports...")
        
        df.to_csv('outputs/data/covid_tweets_annotated.csv', index=False)
        df.to_parquet('outputs/data/covid_tweets_annotated.parquet', index=False)
        
        statistical_summary = self._generate_statistical_summary(df)
        with open('outputs/reports/statistical_summary.json', 'w') as f:
            json.dump(statistical_summary, f, indent=2, default=str)
        
        if 'topics' in self.results:
            topic_report = self._generate_topic_report(df)
            with open('outputs/reports/topic_report.json', 'w') as f:
                json.dump(topic_report, f, indent=2, default=str)
        
        repro_report = self._generate_reproducibility_report()
        with open('outputs/reproducibility/reproducibility_report.json', 'w') as f:
            json.dump(repro_report, f, indent=2, default=str)
        
        logger.info("Reports saved to outputs/reports/")
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict:
        summary = {
            'dataset_statistics': {
                'total_tweets': int(len(df)),
                'unique_users': int(df['original_author'].nunique()) if 'original_author' in df.columns else 'N/A',
                'date_range': {
                    'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else 'N/A',
                    'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else 'N/A'
                },
                'phases': df['phase'].value_counts().to_dict() if 'phase' in df.columns else {}
            }
        }
        
        if 'topic' in df.columns:
            summary['topic_statistics'] = {
                'num_topics': int(df[df['topic'] != -1]['topic'].nunique()),
                'noise_percentage': float((df['topic'] == -1).mean() * 100),
                'avg_topic_size': float(df[df['topic'] != -1]['topic'].value_counts().mean())
            }
        
        if 'sentiment_numeric' in df.columns:
            summary['sentiment_statistics'] = {
                'positive': float((df['sentiment_numeric'] == 1).mean() * 100),
                'neutral': float((df['sentiment_numeric'] == 0).mean() * 100),
                'negative': float((df['sentiment_numeric'] == -1).mean() * 100),
                'avg_sentiment': float(df['sentiment_numeric'].mean())
            }
        
        if 'anomaly_label' in df.columns:
            summary['anomaly_statistics'] = {
                'anomaly_rate': float(df['anomaly_label'].mean() * 100),
                'total_anomalies': int(df['anomaly_label'].sum())
            }
        
        if 'bot_label' in df.columns:
            summary['bot_statistics'] = {
                'bot_tweet_rate': float(df['bot_label'].mean() * 100),
                'total_bot_tweets': int(df['bot_label'].sum())
            }
        
        return summary
    
    def _generate_topic_report(self, df: pd.DataFrame) -> Dict:
        report = {
            'total_topics': self.results['topics']['num_topics'],
            'topics': []
        }
        
        if 'characterization' in self.results['topics']:
            for topic_id, info in self.results['topics']['characterization'].items():
                topic_entry = {
                    'topic_id': int(topic_id),
                    'keywords': info['keywords'][:10] if 'keywords' in info else [],
                    'size': int(info['size']),
                    'percentage': float(info['percentage']),
                    'representative_tweets': info.get('representative_tweets', [])[:3]
                }
                
                if 'affective' in self.results and 'topic_affective' in self.results['affective']:
                    if topic_id in self.results['affective']['topic_affective']:
                        topic_entry['sentiment'] = self.results['affective']['topic_affective'][topic_id]['sentiment']
                
                report['topics'].append(topic_entry)
        
        return report
    
    def _generate_reproducibility_report(self) -> Dict:
        import transformers
        import sentence_transformers
        import sklearn
        
        #TODO(FIX hdbscan version)
        report = {
            'pipeline_version': '1.0',
            'execution_date': str(datetime.now()),
            'random_seed': RANDOM_SEED,
            'configuration': self.config,
            'environment': {
                'python_version': platform.python_version(),
                'platform': platform.platform(),
                'libraries': {
                    'numpy': np.__version__,
                    'pandas': pd.__version__,
                    'torch': torch.__version__,
                    'transformers': transformers.__version__,
                    'sentence_transformers': sentence_transformers.__version__,
                    'sklearn': sklearn.__version__,
                    'umap-learn': umap.__version__,
                    'hdbscan': '0.8.39 (Installed Placeholder)'
                },
                'hardware': {
                    'cpu': platform.processor(),
                    'cpu_count': os.cpu_count(),
                    'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'
                }
            }
        }
        
        return report
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECKPOINT SUPPORT
    # ═══════════════════════════════════════════════════════════════════════
    
    def _save_checkpoint(self, df: pd.DataFrame, phase: int):
        if not self.config['use_checkpoints']:
            return
        
        checkpoint_data = {
            'df': df,
            'phase': phase,
            'timestamp': datetime.now(),
            'results': self.results
        }
        with open('outputs/data/checkpoint.pkl', 'wb') as f:
            pickle.dump(checkpoint_data, f)
        logger.info(f"Checkpoint saved at phase {phase}")
    
    def _load_checkpoint(self):
        checkpoint_file = 'outputs/data/checkpoint.pkl'
        if os.path.exists(checkpoint_file):
            logger.info("Loading from checkpoint...")
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            return checkpoint_data
        return None
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ═══════════════════════════════════════════════════════════════════════
    
    def run_complete_pipeline(self, data_paths: List[str]) -> pd.DataFrame:
        logger.info("="*80)
        logger.info("COVID-19 SOCIAL MEDIA DISCOURSE ANALYSIS PIPELINE")
        logger.info("="*80)
        start_time = datetime.now()
        
        try:
            checkpoint = self._load_checkpoint()
            last_phase = checkpoint['phase'] if checkpoint else 0
            
            if last_phase >= 1 and checkpoint:
                df = checkpoint['df']
                logger.info(f"[DEBUG] df rows: {len(df)}")
                self.results = checkpoint.get('results', {})
                logger.info(f"Resuming from phase {last_phase}")
            else:
                df = self.load_and_preprocess_data(data_paths)
                logger.info(f"[DEBUG] df rows: {len(df)}")
                self._save_checkpoint(df, 1)
                last_phase = 1
            
            if last_phase >= 2 and os.path.exists('outputs/data/intermediate/embeddings.npy'):
                embeddings = np.load('outputs/data/intermediate/embeddings.npy')
                logger.info(f"[DEBUG] df rows: {len(df)}")
                self.embeddings = embeddings
            else:
                embeddings = self.generate_embeddings(df)
                logger.info(f"[DEBUG] df rows: {len(df)}")
                self._save_checkpoint(df, 2)
                last_phase = 2
            
            if last_phase < 3:
                topic_results = self.discover_topics(df, embeddings)
                logger.info(f"[DEBUG] df rows: {len(df)}")
                self._save_checkpoint(df, 3)
            
            if last_phase < 4:
                df = self.analyze_sentiment_emotion(df)
                logger.info(f"[DEBUG] df rows: {len(df)}")
                self._save_checkpoint(df, 4)
            
            if last_phase < 5:
                df = self.detect_anomalies(df, embeddings)
                logger.info(f"[DEBUG] df rows: {len(df)}")
                self._save_checkpoint(df, 5)
            
            if last_phase < 6:
                df = self.detect_bots(df)
                logger.info(f"[DEBUG] df rows: {len(df)}")
                self._save_checkpoint(df, 6)
            
            if last_phase < 7:
                stats_results = self.perform_statistical_analysis(df)
                logger.info(f"[DEBUG] df rows: {len(df)}")
                self._save_checkpoint(df, 7)
            
            self.generate_visualizations(df)
            self.generate_reports(df)
            
            end_time = datetime.now()
            runtime = end_time - start_time
            
            logger.info("="*80)
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info(f"Total tweets processed: {len(df):,}")
            if 'topics' in self.results:
                logger.info(f"Topics identified: {self.results['topics']['num_topics']}")
            if 'anomalies' in self.results:
                logger.info(f"Anomalies detected: {self.results['anomalies']['total_anomalies']:,}")
            if 'bots' in self.results:
                logger.info(f"Bot tweets identified: {self.results['bots']['bot_tweets']:,}")
            logger.info(f"Total runtime: {runtime}")
            logger.info(f"Outputs saved to: outputs/")
            logger.info("="*80)
            
            if os.path.exists('outputs/data/checkpoint.pkl'):
                os.remove('outputs/data/checkpoint.pkl')
            
            return df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    config = {
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'batch_size': 32,
        'chunk_size': 10000,
        'use_checkpoints': True
    }
    
    pipeline = CovidTwitterAnalysisPipeline(config=config)
    
    data_paths = [
        "Covid-19 Twitter Dataset (Apr-Jun 2020).csv",
    ]
    
    try:
        annotated_df = pipeline.run_complete_pipeline(data_paths)
        print("\nPipeline completed successfully!")
        print(f"Annotated dataset shape: {annotated_df.shape}")
        print(f"Columns: {list(annotated_df.columns)}")
    except Exception as e:
        print(f"Error running pipeline: {e}")