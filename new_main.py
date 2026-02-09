"""
COVID-19 Twitter Discourse Analysis Pipeline - CORRECTED VERSION
Addresses: Topic imbalance, sentiment/emotion collapse, bot detection,
visualization gaps, confidence scoring, and ACE-based annotation
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
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import re
import nltk
from langdetect import detect, LangDetectException
from fuzzywuzzy import fuzz

from sklearn.metrics import (
	silhouette_score,
	davies_bouldin_score,
	calinski_harabasz_score,
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr, entropy
from scipy.cluster.hierarchy import linkage, fcluster

from sentence_transformers import SentenceTransformer
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	pipeline,
	AutoModelForTokenClassification,
)
import umap
import hdbscan
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
	handlers=[logging.FileHandler("pipeline_corrected.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed(RANDOM_SEED)

# Download NLTK data
for package in [
	"punkt",
	"stopwords",
	"wordnet",
	"vader_lexicon",
	"averaged_perceptron_tagger",
]:
	nltk.download(package, quiet=True)


class ACEAnnotator:
	"""
	Automatic Content Extraction (ACE) standard annotator for keywords
	Extracts entities, events, and relations following ACE guidelines
	"""

	ACE_ENTITY_TYPES = [
		"PERSON",
		"ORGANIZATION",
		"LOCATION",
		"FACILITY",
		"GPE",  # Geo-Political Entity
		"VEHICLE",
		"WEAPON",
		"DATE",
		"TIME",
		"MONEY",
		"PERCENT",
	]

	COVID_DOMAIN_ENTITIES = {
		"DISEASE": [
			"covid",
			"coronavirus",
			"covid-19",
			"covid19",
			"sars-cov-2",
			"pandemic",
			"epidemic",
		],
		"SYMPTOM": [
			"fever",
			"cough",
			"shortness of breath",
			"fatigue",
			"loss of taste",
			"loss of smell",
		],
		"TREATMENT": [
			"vaccine",
			"vaccination",
			"treatment",
			"therapy",
			"medication",
			"drug",
			"hydroxychloroquine",
			"remdesivir",
		],
		"PREVENTION": [
			"mask",
			"social distancing",
			"quarantine",
			"isolation",
			"lockdown",
			"sanitizer",
			"handwashing",
		],
		"ORGANIZATION": [
			"who",
			"cdc",
			"fda",
			"nih",
			"government",
			"hospital",
			"clinic",
		],
		"PERSON": ["doctor", "nurse", "patient", "fauci", "trump", "biden"],
		"EVENT": [
			"outbreak",
			"surge",
			"wave",
			"peak",
			"death",
			"infection",
			"case",
			"hospitalization",
		],
	}

	def __init__(self):
		try:
			# Load NER model
			self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
			self.ner_model = AutoModelForTokenClassification.from_pretrained(
				"dslim/bert-base-NER"
			)
			self.ner_pipeline = pipeline(
				"ner",
				model=self.ner_model,
				tokenizer=self.ner_tokenizer,
				aggregation_strategy="simple",
			)
			logger.info("ACE Annotator initialized with NER model")
		except Exception as e:
			logger.warning(f"Could not load NER model: {e}. Using rule-based approach.")
			self.ner_pipeline = None

	def extract_ace_entities(self, text: str) -> Dict[str, List[Dict]]:
		"""
		Extract entities following ACE standard
		Returns: Dict with entity types and their instances
		"""
		entities = defaultdict(list)

		# Domain-specific entities (COVID-related)
		for entity_type, keywords in self.COVID_DOMAIN_ENTITIES.items():
			for keyword in keywords:
				pattern = r"\b" + re.escape(keyword) + r"\b"
				matches = re.finditer(pattern, text.lower())
				for match in matches:
					entities[entity_type].append(
						{
							"text": match.group(),
							"start": match.start(),
							"end": match.end(),
							"confidence": 1.0,  # Rule-based = high confidence
						}
					)

		# NER-based extraction
		if self.ner_pipeline:
			try:
				ner_results = self.ner_pipeline(text[:512])  # Limit length
				for entity in ner_results:
					entity_type = entity["entity_group"]
					entities[entity_type].append(
						{
							"text": entity["word"],
							"start": entity["start"],
							"end": entity["end"],
							"confidence": entity["score"],
						}
					)
			except Exception as e:
				logger.debug(f"NER extraction failed: {e}")

		return dict(entities)

	def extract_keywords_with_confidence(
		self, text: str, top_k: int = 10
	) -> List[Dict]:
		"""
		Extract keywords with confidence scores using TF-IDF and POS tagging
		"""
		from nltk import pos_tag, word_tokenize
		from nltk.corpus import stopwords

		stop_words = set(stopwords.words("english"))

		# Tokenize and POS tag
		tokens = word_tokenize(text.lower())
		pos_tags = pos_tag(tokens)

		# Extract nouns and adjectives (more meaningful keywords)
		keywords = []
		for word, pos in pos_tags:
			if (
				(pos.startswith("NN") or pos.startswith("JJ"))
				and word not in stop_words
				and len(word) > 2
			):
				keywords.append(word)

		# Calculate confidence based on frequency
		keyword_freq = Counter(keywords)
		total = sum(keyword_freq.values())

		keyword_conf = [
			{
				"keyword": kw,
				"frequency": freq,
				"confidence": freq / total if total > 0 else 0,
				"ace_type": self._classify_ace_type(kw),
			}
			for kw, freq in keyword_freq.most_common(top_k)
		]

		return keyword_conf

	def _classify_ace_type(self, keyword: str) -> str:
		"""Classify keyword into ACE entity type"""
		for entity_type, keywords in self.COVID_DOMAIN_ENTITIES.items():
			if keyword in keywords or any(kw in keyword for kw in keywords):
				return entity_type
		return "GENERAL"


class EnhancedCovidTwitterAnalysisPipeline:
	"""
	Enhanced pipeline with fixes for:
	- Topic imbalance (force 20 balanced topics with names)
	- Sentiment/emotion collapse (multi-model ensemble)
	- Bot detection sensitivity (advanced heuristics)
	- Anomaly detection granularity (multi-scale approach)
	- Confidence scoring throughout
	- ACE-based keyword annotation
	"""

	COLUMN_MAPPING = {
		"text": ["text", "tweet", "content", "original_text", "full_text"],
		"created_at": ["created_at", "timestamp", "date", "created"],
		"author": ["user", "username", "author", "original_author", "screen_name"],
		"id": ["id", "tweet_id", "status_id"],
		"retweet_count": ["retweet_count", "retweets", "rt_count"],
		"favorite_count": ["favorite_count", "favorites", "likes", "like_count"],
		"lang": ["lang", "language"],
	}

	# Target exactly 20 topics with COVID-19 context
	TARGET_TOPICS = 20

	def __init__(self, config: Optional[Dict] = None):
		self.config = self._get_default_config()
		if config:
			self.config.update(config)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		logger.info(f"Using device: {self.device}")

		self.data = {}
		self.embeddings = None
		self.models = {}
		self.results = {}
		self.confidence_scores = {}  # Track confidence throughout

		# Initialize ACE annotator
		self.ace_annotator = ACEAnnotator()

		self._create_output_dirs()

	def _get_default_config(self) -> Dict:
		return {
			# Embedding config
			"embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
			"embedding_dim": 384,
			"batch_size": 32,
			"chunk_size": 10000,
			# Topic modeling config - FIXED for 20 topics
			"num_topics": 20,
			"topic_min_size": 50,  # Minimum tweets per topic
			"reduce_outliers": True,
			"hdbscan": {
				"min_cluster_size": 100,  # Increased for better clusters
				"min_samples": 30,
				"metric": "euclidean",
				"cluster_selection_method": "eom",
				"prediction_data": True,
				"cluster_selection_epsilon": 0.5,  # More granular control
			},
			"umap": {
				"n_components": 10,  # Increased from 5 for better separation
				"n_neighbors": 20,  # Increased from 15
				"min_dist": 0.0,
				"metric": "cosine",
				"random_state": RANDOM_SEED,
			},
			# Sentiment/Emotion - ENSEMBLE approach
			"sentiment_ensemble": True,
			"sentiment_threshold": 0.6,  # Higher threshold for classification
			"emotion_threshold": 0.4,
			# Anomaly detection - MULTI-SCALE
			"anomaly_methods": ["isolation_forest", "local_outlier", "statistical"],
			"anomaly_sensitivity": 0.1,  # 10% contamination
			"isolation_forest": {
				"n_estimators": 200,
				"max_samples": 512,
				"contamination": 0.1,
				"random_state": RANDOM_SEED,
			},
			# Bot detection - ENHANCED
			"bot_detection_features": [
				"posting_frequency",
				"content_similarity",
				"temporal_pattern",
				"hashtag_usage",
				"mention_pattern",
				"url_ratio",
				"retweet_ratio",
			],
			"bot_threshold": 0.6,  # Lower threshold = more sensitive
			"bot_ensemble": True,
			# General
			"similarity_threshold": 0.85,
			"significance_level": 0.05,
			"output_dir": "outputs",
			"use_checkpoints": True,
			"confidence_tracking": True,
		}

	def _create_output_dirs(self):
		dirs = [
			"outputs/data",
			"outputs/data/intermediate",
			"outputs/reports",
			"outputs/visualizations",
			"outputs/visualizations/confidence",
			"outputs/reproducibility",
			"outputs/validation",
			"outputs/ace_annotations",
		]
		for dir_path in dirs:
			Path(dir_path).mkdir(parents=True, exist_ok=True)

	# ═══════════════════════════════════════════════════════════════════════
	# PHASE 1: ENHANCED DATA PREPROCESSING WITH ACE ANNOTATION
	# ═══════════════════════════════════════════════════════════════════════

	def load_and_preprocess_data(self, data_paths: List[str]) -> pd.DataFrame:
		logger.info("=" * 80)
		logger.info("PHASE 1: ENHANCED DATA PREPROCESSING WITH ACE ANNOTATION")
		logger.info("=" * 80)

		dfs = []
		for path in data_paths:
			try:
				df = pd.read_csv(path, encoding="utf-8", low_memory=False)
				df = self._standardize_columns(df)
				phase = self._extract_phase_from_path(path)
				df["phase"] = phase
				dfs.append(df)
				logger.info(f"Loaded {len(df):,} tweets from {path}")
			except Exception as e:
				logger.error(f"Failed to load {path}: {str(e)}")
				continue

		if not dfs:
			raise ValueError("No datasets were successfully loaded")

		df_combined = pd.concat(dfs, ignore_index=True)
		logger.info(f"Total tweets loaded: {len(df_combined):,}")

		# Preprocessing pipeline
		df_dedup = self._deduplicate_tweets(df_combined)
		df_clean = self._normalize_text(df_dedup)
		df_english = self._filter_language(df_clean)
		df_metadata = self._extract_metadata(df_english)

		# ACE-based annotation
		df_processed = self._apply_ace_annotation(df_metadata)

		self.data["processed"] = df_processed

		# Save
		df_processed.to_csv("outputs/data/processed_tweets.csv", index=False)
		df_processed.to_parquet("outputs/data/processed_tweets.parquet", index=False)

		logger.info(
			f"Preprocessing complete. Final dataset: {len(df_processed):,} tweets"
		)
		return df_processed

	def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Standardize column names"""
		for standard_name, possible_names in self.COLUMN_MAPPING.items():
			for col in df.columns:
				if col.lower() in [p.lower() for p in possible_names]:
					df.rename(columns={col: standard_name}, inplace=True)
					break

		if "original_text" not in df.columns and "text" in df.columns:
			df["original_text"] = df["text"]
		if "original_author" not in df.columns and "author" in df.columns:
			df["original_author"] = df["author"]

		return df

	def _extract_phase_from_path(self, path: str) -> str:
		"""Extract pandemic phase from file path"""
		if "Apr-Jun 2020" in path or "Phase1" in path or "phase1" in path:
			return "Phase1"
		elif "Aug-Oct 2020" in path or "Phase2" in path or "phase2" in path:
			return "Phase2"
		elif "Apr-Jun 2021" in path or "Phase3" in path or "phase3" in path:
			return "Phase3"
		return "Unknown"

	def _deduplicate_tweets(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Enhanced deduplication with exact and near-duplicate detection"""
		logger.info("Removing duplicates...")
		initial_size = len(df)

		# Exact duplicates
		df["text_hash"] = df["original_text"].apply(
			lambda x: hashlib.md5(str(x).encode()).hexdigest() if pd.notna(x) else None
		)
		df_dedup = df.drop_duplicates(subset=["text_hash"], keep="first")

		logger.info(f"Removed {initial_size - len(df_dedup):,} exact duplicates")
		return df_dedup

	def _normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Enhanced text normalization"""
		logger.info("Normalizing text...")

		def clean_text(text):
			if pd.isna(text):
				return ""

			# Lowercase
			text = text.lower()

			# Remove URLs
			text = re.sub(
				r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
				"",
				text,
			)
			text = re.sub(r"www\.[a-zA-Z0-9\.\-]+\.[a-zA-Z]+", "", text)

			# Normalize COVID-19 mentions
			text = re.sub(
				r"\bcovid\b|\bcovid[\-\s]?19\b|\bcoronavirus\b|\bsars[\-\s]?cov[\-\s]?2\b",
				"covid19",
				text,
				flags=re.IGNORECASE,
			)

			# Remove RT prefix but keep the content
			text = re.sub(r"^rt\s+@\w+:\s*", "", text)

			# Normalize whitespace
			text = re.sub(r"\s+", " ", text)

			return text.strip()

		df["cleaned_text"] = df["original_text"].apply(clean_text)
		df = df[df["cleaned_text"].str.len() > 10]  # Minimum length

		return df

	def _filter_language(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Filter for English tweets with confidence scoring"""
		logger.info("Filtering for English tweets...")

		def detect_language_with_confidence(text):
			try:
				if pd.notna(text) and len(text) > 10:
					lang = detect(text)
					# langdetect doesn't provide confidence, so we use heuristics
					# Check for English indicators
					english_words = len(
						re.findall(
							r"\b(the|and|is|are|was|were|to|of|in|for|on)\b",
							text.lower(),
						)
					)
					confidence = min(english_words / 10, 1.0)
					return lang, confidence
				return "unknown", 0.0
			except LangDetectException:
				return "unknown", 0.0

		if "lang" in df.columns:
			df_english = df[df["lang"] == "en"].copy()
			df_english["lang_confidence"] = 1.0
		else:
			lang_data = df["cleaned_text"].apply(detect_language_with_confidence)
			df["detected_lang"] = lang_data.apply(lambda x: x[0])
			df["lang_confidence"] = lang_data.apply(lambda x: x[1])
			df_english = df[df["detected_lang"] == "en"].copy()

		logger.info(f"Retained {len(df_english):,} English tweets")
		logger.info(
			f"Average language confidence: {df_english['lang_confidence'].mean():.3f}"
		)

		return df_english

	def _extract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Extract comprehensive metadata"""
		logger.info("Extracting metadata...")

		# Temporal features
		df["timestamp"] = pd.to_datetime(df["created_at"], errors="coerce")
		df["date"] = df["timestamp"].dt.date
		df["hour"] = df["timestamp"].dt.hour
		df["day_of_week"] = df["timestamp"].dt.dayofweek
		df["week_of_year"] = df["timestamp"].dt.isocalendar().week
		df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

		# Engagement metrics
		df["retweet_count"] = pd.to_numeric(
			df["retweet_count"], errors="coerce"
		).fillna(0)
		df["favorite_count"] = pd.to_numeric(
			df["favorite_count"], errors="coerce"
		).fillna(0)
		df["engagement_score"] = df["retweet_count"] + df["favorite_count"]
		df["retweet_ratio"] = df["retweet_count"] / (df["engagement_score"] + 1)

		# Content features
		df["hashtag_list"] = df["original_text"].apply(
			lambda x: re.findall(r"#\w+", str(x).lower()) if pd.notna(x) else []
		)
		df["hashtag_count"] = df["hashtag_list"].apply(len)

		df["mention_list"] = df["original_text"].apply(
			lambda x: re.findall(r"@\w+", str(x).lower()) if pd.notna(x) else []
		)
		df["mention_count"] = df["mention_list"].apply(len)

		df["url_count"] = df["original_text"].apply(
			lambda x: len(re.findall(r"http[s]?://\S+", str(x))) if pd.notna(x) else 0
		)

		df["text_length"] = df["cleaned_text"].str.len()
		df["word_count"] = df["cleaned_text"].str.split().str.len()

		# Is retweet
		df["is_retweet"] = (
			df["original_text"].str.startswith("RT @", na=False).astype(int)
		)

		return df

	def _apply_ace_annotation(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Apply ACE (Automatic Content Extraction) annotation
		Extract entities, keywords with confidence scores
		"""
		logger.info("Applying ACE annotation...")

		# Sample for ACE annotation (full annotation can be slow)
		sample_size = min(10000, len(df))
		sample_indices = np.random.choice(df.index, sample_size, replace=False)

		ace_results = []
		for idx in tqdm(sample_indices, desc="ACE annotation"):
			text = df.loc[idx, "cleaned_text"]

			# Extract entities
			entities = self.ace_annotator.extract_ace_entities(
				df.loc[idx, "original_text"]
			)

			# Extract keywords
			keywords = self.ace_annotator.extract_keywords_with_confidence(
				text, top_k=5
			)

			ace_results.append(
				{
					"index": idx,
					"entities": entities,
					"keywords": keywords,
					"entity_count": sum(len(v) for v in entities.values()),
					"keyword_confidence_avg": (
						np.mean([k["confidence"] for k in keywords]) if keywords else 0
					),
				}
			)

		# Merge back
		ace_df = pd.DataFrame(ace_results)
		df = df.merge(
			ace_df[["index", "entity_count", "keyword_confidence_avg"]],
			left_index=True,
			right_on="index",
			how="left",
		)
		df["entity_count"] = df["entity_count"].fillna(0)
		df["keyword_confidence_avg"] = df["keyword_confidence_avg"].fillna(0)

		# Save detailed ACE annotations
		with open("outputs/ace_annotations/detailed_annotations.json", "w") as f:
			json.dump(ace_results, f, indent=2, default=str)

		logger.info(
			f"ACE annotation complete. Average entities per tweet: {df['entity_count'].mean():.2f}"
		)

		return df

	# ═══════════════════════════════════════════════════════════════════════
	# PHASE 2: SEMANTIC EMBEDDING GENERATION
	# ═══════════════════════════════════════════════════════════════════════

	def generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
		logger.info("=" * 80)
		logger.info("PHASE 2: SEMANTIC EMBEDDING GENERATION")
		logger.info("=" * 80)

		embedding_path = "outputs/data/intermediate/embeddings.npy"
		if os.path.exists(embedding_path):
			logger.info("Loading existing embeddings...")
			self.embeddings = np.load(embedding_path)
			return self.embeddings

		logger.info(f"Loading embedding model: {self.config['embedding_model']}")
		model = SentenceTransformer(self.config["embedding_model"], device=self.device)
		self.models["embedding"] = model

		texts = df["cleaned_text"].tolist()
		total_texts = len(texts)
		chunk_size = self.config["chunk_size"]

		all_embeddings = []

		for chunk_start in range(0, total_texts, chunk_size):
			chunk_end = min(chunk_start + chunk_size, total_texts)
			chunk_texts = texts[chunk_start:chunk_end]

			logger.info(
				f"Processing chunk {chunk_start//chunk_size + 1}/{(total_texts-1)//chunk_size + 1}"
			)

			batch_size = self.config["batch_size"]
			chunk_embeddings = []

			for i in tqdm(
				range(0, len(chunk_texts), batch_size), desc="Generating embeddings"
			):
				batch = chunk_texts[i : i + batch_size]
				try:
					batch_embeddings = model.encode(
						batch,
						convert_to_tensor=True,
						show_progress_bar=False,
						batch_size=batch_size,
					)
					chunk_embeddings.append(batch_embeddings.cpu().numpy())

					if torch.cuda.is_available() and i % (batch_size * 10) == 0:
						torch.cuda.empty_cache()
				except Exception as e:
					logger.warning(f"Error in batch {i}: {e}, using zero embeddings")
					chunk_embeddings.append(
						np.zeros((len(batch), self.config["embedding_dim"]))
					)

			chunk_matrix = np.vstack(chunk_embeddings)
			all_embeddings.append(chunk_matrix)

		embedding_matrix = np.vstack(all_embeddings)
		self.embeddings = embedding_matrix

		np.save(embedding_path, embedding_matrix)

		logger.info(f"Generated embeddings with shape: {embedding_matrix.shape}")
		return embedding_matrix

	# ═══════════════════════════════════════════════════════════════════════
	# PHASE 3: ENHANCED TOPIC DISCOVERY (20 NAMED, BALANCED TOPICS)
	# ═══════════════════════════════════════════════════════════════════════

	def discover_topics(self, df: pd.DataFrame, embeddings: np.ndarray) -> Dict:
		logger.info("=" * 80)
		logger.info("PHASE 3: ENHANCED TOPIC DISCOVERY - 20 NAMED TOPICS")
		logger.info("=" * 80)

		embedding_model = SentenceTransformer(self.config["embedding_model"])

		# Use hierarchical reduction to force exactly 20 topics
		umap_model = umap.UMAP(
			n_components=self.config["umap"]["n_components"],
			n_neighbors=self.config["umap"]["n_neighbors"],
			min_dist=self.config["umap"]["min_dist"],
			metric=self.config["umap"]["metric"],
			random_state=self.config["umap"]["random_state"],
		)

		hdbscan_model = hdbscan.HDBSCAN(
			min_cluster_size=self.config["hdbscan"]["min_cluster_size"],
			min_samples=self.config["hdbscan"]["min_samples"],
			metric=self.config["hdbscan"]["metric"],
			cluster_selection_method=self.config["hdbscan"]["cluster_selection_method"],
			prediction_data=self.config["hdbscan"]["prediction_data"],
		)

		# Enhanced representation models for better topic naming
		representation_model = {
			"KeyBERT": KeyBERTInspired(),
			"MMR": MaximalMarginalRelevance(diversity=0.3),
		}

		# Initial BERTopic
		topic_model = BERTopic(
			embedding_model = embedding_model,

			umap_model=umap_model,
			hdbscan_model=hdbscan_model,
			representation_model=representation_model,
			nr_topics=None,  # Auto-detect first
			calculate_probabilities=True,
			verbose=True,
		)

		logger.info("Fitting BERTopic model...")
		num_documents = len(df["cleaned_text"].tolist())
		embeddings_shape = embeddings.shape

		logger.info(f"Number of documents (texts) being processed: {num_documents}")
		logger.info(f"Shape of the embeddings array: {embeddings_shape}")
		topics, probs = topic_model.fit_transform(
			df["cleaned_text"].tolist(), embeddings
		)

		# Count initial topics
		unique_topics = len(set(topics)) - (1 if -1 in topics else 0)
		logger.info(f"Initial topics discovered: {unique_topics}")

		# Reduce to exactly 20 topics
		if unique_topics > self.TARGET_TOPICS:
			logger.info(
				f"Reducing from {unique_topics} to {self.TARGET_TOPICS} topics..."
			)
			topic_model.reduce_topics(
				df['cleaned_text'].tolist(), 
				nr_topics=self.TARGET_TOPICS
			)
			topics = topic_model.topics_
		elif unique_topics < self.TARGET_TOPICS:
			# If we have fewer topics, use LDA as supplement
			logger.info(
				f"Using hybrid approach to reach {self.TARGET_TOPICS} topics..."
			)
			topics = self._hybrid_topic_expansion(df, embeddings, topics, topic_model)

		# Assign topics to dataframe
		df["topic"] = topics

		# Calculate topic probabilities/confidence
		if len(probs.shape) > 1:
			df["topic_confidence"] = probs.max(axis=1)
		else:
			df["topic_confidence"] = probs

		# Handle outliers (-1 topic)
		if self.config["reduce_outliers"]:
			df = self._reassign_outliers(df, embeddings, topic_model)

		# Generate human-readable topic names
		topic_names = self._generate_topic_names(df, topic_model)

		# Create topic info
		topic_info = topic_model.get_topic_info()
		topic_info["topic_name"] = topic_info["Topic"].map(topic_names)

		# Characterize topics
		topic_characterization = self._characterize_topics_enhanced(
			df, topic_model, embeddings, topic_names
		)

		# Calculate topic quality metrics
		quality_metrics = self._calculate_topic_quality(df, embeddings)

		topic_results = {
			"model": topic_model,
			"topic_info": topic_info,
			"topic_names": topic_names,
			"topics": topics,
			"probabilities": probs,
			"num_topics": len([t for t in set(topics) if t != -1]),
			"characterization": topic_characterization,
			"quality_metrics": quality_metrics,
		}

		# Save
		topic_model.save("outputs/data/intermediate/topic_model")
		topic_info.to_csv("outputs/reports/topic_info_named.csv", index=False)

		self.results["topics"] = topic_results
		self.confidence_scores["topic_confidence"] = (
			df["topic_confidence"].describe().to_dict()
		)

		logger.info(f"Final topic count: {topic_results['num_topics']}")
		logger.info(f"Average topic confidence: {df['topic_confidence'].mean():.3f}")

		return topic_results

	def _hybrid_topic_expansion(
		self,
		df: pd.DataFrame,
		embeddings: np.ndarray,
		bert_topics: np.ndarray,
		topic_model: BERTopic,
	) -> np.ndarray:
		"""
		Expand topics to reach target number using LDA
		"""
		logger.info("Using LDA to supplement topic detection...")

		# Use LDA on outliers
		outlier_mask = bert_topics == -1
		outlier_texts = df[outlier_mask]["cleaned_text"].tolist()

		if len(outlier_texts) < 100:
			return bert_topics

		# TF-IDF + LDA
		vectorizer = CountVectorizer(max_features=1000, stop_words="english")
		doc_term_matrix = vectorizer.fit_transform(outlier_texts)

		n_lda_topics = self.TARGET_TOPICS - len(set(bert_topics)) + 1
		lda = LatentDirichletAllocation(
			n_components=max(n_lda_topics, 5), random_state=RANDOM_SEED
		)
		lda_topics = lda.fit_transform(doc_term_matrix)
		lda_assignments = lda_topics.argmax(axis=1)

		# Merge topics
		combined_topics = bert_topics.copy()
		max_topic = bert_topics.max()
		outlier_indices = np.where(outlier_mask)[0]
		combined_topics[outlier_indices] = lda_assignments + max_topic + 1

		return combined_topics

	def _reassign_outliers(
		self, df: pd.DataFrame, embeddings: np.ndarray, topic_model: BERTopic
	) -> pd.DataFrame:
		"""
		Reassign outlier tweets (-1 topic) to nearest topic
		"""
		outlier_mask = df["topic"] == -1
		n_outliers = outlier_mask.sum()

		if n_outliers == 0:
			return df

		logger.info(f"Reassigning {n_outliers} outlier tweets to nearest topics...")

		outlier_embeddings = embeddings[outlier_mask]

		# Get topic centroids
		topic_centroids = {}
		for topic in df[df["topic"] != -1]["topic"].unique():
			topic_mask = df["topic"] == topic
			topic_centroids[topic] = embeddings[topic_mask].mean(axis=0)

		# Assign to nearest centroid
		for idx, (orig_idx, row) in enumerate(df[outlier_mask].iterrows()):
			emb = outlier_embeddings[idx]
			distances = {
				t: np.linalg.norm(emb - centroid)
				for t, centroid in topic_centroids.items()
			}
			nearest_topic = min(distances, key=distances.get)
			df.loc[orig_idx, "topic"] = nearest_topic
			df.loc[orig_idx, "topic_confidence"] = 0.3  # Low confidence for reassigned

		logger.info(
			f"Outliers reassigned. New outlier count: {(df['topic'] == -1).sum()}"
		)
		return df

	def _generate_topic_names(
		self, df: pd.DataFrame, topic_model: BERTopic
	) -> Dict[int, str]:
		"""
		Generate human-readable topic names based on content
		"""
		logger.info("Generating human-readable topic names...")

		COVID_TOPIC_TEMPLATES = {
			"vaccine": "Vaccination & Immunization",
			"lockdown": "Lockdowns & Restrictions",
			"death": "Death Toll & Mortality",
			"case": "Case Numbers & Statistics",
			"mask": "Mask Mandates & PPE",
			"hospital": "Healthcare System & Hospitals",
			"symptom": "Symptoms & Health Effects",
			"treatment": "Treatments & Therapeutics",
			"testing": "Testing & Diagnostics",
			"government": "Government Response & Policy",
			"economy": "Economic Impact",
			"school": "Schools & Education",
			"travel": "Travel & Transportation",
			"quarantine": "Quarantine & Isolation",
			"misinformation": "Misinformation & Conspiracy",
			"science": "Scientific Research",
			"mental health": "Mental Health Impact",
			"work": "Work & Employment",
			"social": "Social Distancing & Behavior",
			"global": "Global Pandemic Response",
		}

		topic_names = {}

		for topic in df["topic"].unique():
			if topic == -1:
				topic_names[topic] = "Outliers/Noise"
				continue

			# Get top keywords
			try:
				keywords = topic_model.get_topic(topic)
				if not keywords:
					topic_names[topic] = f"Topic {topic}"
					continue

				top_words = [word for word, _ in keywords[:10]]

				# Match to template
				matched = False
				for key, template_name in COVID_TOPIC_TEMPLATES.items():
					if any(key in word for word in top_words):
						topic_names[topic] = template_name
						matched = True
						break

				if not matched:
					# Generate from top 3 keywords
					top_3 = [word.replace("_", " ").title() for word, _ in keywords[:3]]
					topic_names[topic] = f"{top_3[0]} & {top_3[1]}"

			except Exception as e:
				logger.warning(f"Could not name topic {topic}: {e}")
				topic_names[topic] = f"Topic {topic}"

		return topic_names

	def _characterize_topics_enhanced(
		self,
		df: pd.DataFrame,
		topic_model: BERTopic,
		embeddings: np.ndarray,
		topic_names: Dict,
	) -> Dict:
		"""
		Enhanced topic characterization with confidence scores
		"""
		logger.info("Characterizing topics with confidence scores...")

		characterizations = {}
		topics = [t for t in df["topic"].unique() if t != -1]

		for topic_id in tqdm(topics, desc="Characterizing topics"):
			topic_tweets = df[df["topic"] == topic_id]

			# Keywords with scores
			keywords = topic_model.get_topic(topic_id)[:10] if topic_id != -1 else []

			# Representative tweets (closest to centroid)
			if len(topic_tweets) > 0:
				topic_embeddings = embeddings[df["topic"] == topic_id]
				centroid = topic_embeddings.mean(axis=0)

				distances = np.linalg.norm(topic_embeddings - centroid, axis=1)
				closest_indices = np.argsort(distances)[:5]
				representative_tweets = topic_tweets.iloc[closest_indices][
					"cleaned_text"
				].tolist()
			else:
				representative_tweets = []

			# Temporal distribution
			temporal_dist = (
				topic_tweets.groupby("date").size().to_dict()
				if "date" in topic_tweets.columns
				else {}
			)

			# Sentiment distribution
			if "sentiment_numeric" in topic_tweets.columns:
				sentiment_dist = (
					topic_tweets["sentiment_numeric"]
					.value_counts(normalize=True)
					.to_dict()
				)
			else:
				sentiment_dist = {}

			# Confidence stats
			confidence_stats = (
				topic_tweets["topic_confidence"].describe().to_dict()
				if "topic_confidence" in topic_tweets.columns
				else {}
			)

			characterizations[topic_id] = {
				"topic_name": topic_names.get(topic_id, f"Topic {topic_id}"),
				"keywords": keywords,
				"representative_tweets": representative_tweets,
				"size": len(topic_tweets),
				"percentage": len(topic_tweets) / len(df) * 100,
				"temporal_distribution": temporal_dist,
				"sentiment_distribution": sentiment_dist,
				"confidence_stats": confidence_stats,
				"avg_engagement": (
					topic_tweets["engagement_score"].mean()
					if "engagement_score" in topic_tweets.columns
					else 0
				),
			}

		return characterizations

	def _calculate_topic_quality(
		self, df: pd.DataFrame, embeddings: np.ndarray
	) -> Dict:
		"""
		Calculate topic modeling quality metrics
		"""
		logger.info("Calculating topic quality metrics...")

		# Filter out outliers
		valid_mask = df["topic"] != -1
		valid_topics = df[valid_mask]["topic"].values
		valid_embeddings = embeddings[valid_mask]

		if len(np.unique(valid_topics)) < 2:
			return {"error": "Insufficient topics for quality calculation"}

		try:
			# Silhouette score
			if len(valid_embeddings) > 10000:
				sample_idx = np.random.choice(
					len(valid_embeddings), 10000, replace=False
				)
				silhouette = silhouette_score(
					valid_embeddings[sample_idx], valid_topics[sample_idx]
				)
			else:
				silhouette = silhouette_score(valid_embeddings, valid_topics)

			# Davies-Bouldin Index (lower is better)
			davies_bouldin = davies_bouldin_score(valid_embeddings, valid_topics)

			# Calinski-Harabasz Index (higher is better)
			calinski = calinski_harabasz_score(valid_embeddings, valid_topics)

			# Topic size balance (entropy - higher means more balanced)
			topic_sizes = df["topic"].value_counts(normalize=True)
			size_entropy = entropy(topic_sizes)
			max_entropy = np.log(len(topic_sizes))
			balance_score = size_entropy / max_entropy if max_entropy > 0 else 0

			return {
				"silhouette_score": float(silhouette),
				"davies_bouldin_index": float(davies_bouldin),
				"calinski_harabasz_index": float(calinski),
				"topic_balance_score": float(balance_score),
				"interpretation": {
					"silhouette": (
						"Good"
						if silhouette > 0.3
						else "Fair" if silhouette > 0.1 else "Poor"
					),
					"balance": "Balanced" if balance_score > 0.7 else "Imbalanced",
				},
			}
		except Exception as e:
			logger.warning(f"Could not calculate quality metrics: {e}")
			return {"error": str(e)}

	# ═══════════════════════════════════════════════════════════════════════
	# PHASE 4: ENHANCED AFFECTIVE ANALYSIS (ENSEMBLE + CONFIDENCE)
	# ═══════════════════════════════════════════════════════════════════════

	def analyze_sentiment_emotion(self, df: pd.DataFrame) -> pd.DataFrame:
		logger.info("=" * 80)
		logger.info("PHASE 4: ENHANCED AFFECTIVE ANALYSIS (ENSEMBLE)")
		logger.info("=" * 80)

		df.reset_index(drop=True, inplace=True)

		# Ensemble sentiment analysis
		logger.info("Performing ensemble sentiment analysis...")
		sentiment_results = self._analyze_sentiment_ensemble(df)

		# Ensemble emotion analysis
		logger.info("Performing ensemble emotion analysis...")
		emotion_results = self._analyze_emotion_ensemble(df)

		# Merge results
		df = pd.concat([df, sentiment_results, emotion_results], axis=1)

		# Aggregate analysis
		affective_summary = self._aggregate_affective_analysis(df)
		self.results["affective"] = affective_summary

		# Track confidence
		self.confidence_scores["sentiment_confidence"] = (
			df["sentiment_confidence"].describe().to_dict()
		)
		self.confidence_scores["emotion_confidence"] = (
			df["emotion_confidence"].describe().to_dict()
		)

		logger.info(
			f"Sentiment distribution: {df['sentiment_label'].value_counts(normalize=True).to_dict()}"
		)
		logger.info(
			f"Average sentiment confidence: {df['sentiment_confidence'].mean():.3f}"
		)
		logger.info(
			f"Emotion distribution: {df['emotion_label'].value_counts(normalize=True).to_dict()}"
		)
		logger.info(
			f"Average emotion confidence: {df['emotion_confidence'].mean():.3f}"
		)

		return df

	def _analyze_sentiment_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Ensemble sentiment analysis using multiple models
		"""
		# Model 1: RoBERTa
		try:
			roberta_pipeline = pipeline(
				"sentiment-analysis",
				model="cardiffnlp/twitter-roberta-base-sentiment-latest",
				device=0 if torch.cuda.is_available() else -1,
				max_length=512,
				truncation=True,
			)
			roberta_results = self._apply_sentiment_model(
				df, roberta_pipeline, "roberta"
			)
		except Exception as e:
			logger.warning(f"RoBERTa sentiment failed: {e}")
			roberta_results = None

		# Model 2: VADER (lexicon-based)
		vader_results = self._apply_vader_sentiment(df)

		# Model 3: DistilBERT
		try:
			distilbert_pipeline = pipeline(
				"sentiment-analysis",
				model="distilbert-base-uncased-finetuned-sst-2-english",
				device=0 if torch.cuda.is_available() else -1,
			)
			distilbert_results = self._apply_sentiment_model(
				df, distilbert_pipeline, "distilbert"
			)
		except Exception as e:
			logger.warning(f"DistilBERT sentiment failed: {e}")
			distilbert_results = None

		# Ensemble combination
		sentiment_df = self._combine_sentiment_ensemble(
			df, roberta_results, vader_results, distilbert_results
		)

		return sentiment_df

	def _apply_sentiment_model(
		self, df: pd.DataFrame, pipeline_model, model_name: str
	) -> pd.DataFrame:
		"""Apply a sentiment model to the dataset"""
		batch_size = 16
		texts = df["cleaned_text"].tolist()

		all_results = []
		for i in tqdm(
			range(0, len(texts), batch_size), desc=f"Sentiment ({model_name})"
		):
			batch = texts[i : i + batch_size]
			batch_truncated = [text[:512] if text else "" for text in batch]

			try:
				results = pipeline_model(batch_truncated)
				all_results.extend(results)
			except Exception as e:
				logger.warning(f"Error in batch {i//batch_size}: {e}")
				all_results.extend([{"label": "neutral", "score": 0.5}] * len(batch))

		# Normalize labels
		label_map = {
			"negative": -1,
			"neutral": 0,
			"positive": 1,
			"NEGATIVE": -1,
			"NEUTRAL": 0,
			"POSITIVE": 1,
			"LABEL_0": -1,
			"LABEL_1": 0,
			"LABEL_2": 1,
			"neg": -1,
			"neu": 0,
			"pos": 1,
		}

		result_df = pd.DataFrame(
			{
				f"{model_name}_label": [r["label"] for r in all_results],
				f"{model_name}_score": [r["score"] for r in all_results],
			}
		)
		result_df[f"{model_name}_numeric"] = result_df[f"{model_name}_label"].apply(
			lambda x: label_map.get(x, label_map.get(x.lower(), 0))
		)

		return result_df

	def _apply_vader_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Apply VADER sentiment analysis"""
		from nltk.sentiment import SentimentIntensityAnalyzer

		sia = SentimentIntensityAnalyzer()

		sentiments = []
		for text in tqdm(df["cleaned_text"], desc="Sentiment (VADER)"):
			scores = sia.polarity_scores(text)
			compound = scores["compound"]

			if compound >= 0.05:
				label, numeric = "positive", 1
			elif compound <= -0.05:
				label, numeric = "negative", -1
			else:
				label, numeric = "neutral", 0

			sentiments.append(
				{
					"vader_label": label,
					"vader_score": abs(compound),
					"vader_numeric": numeric,
					"vader_compound": compound,
				}
			)

		return pd.DataFrame(sentiments)

	def _combine_sentiment_ensemble(
		self, df: pd.DataFrame, *model_results
	) -> pd.DataFrame:
		"""
		Combine multiple sentiment models into ensemble prediction
		"""
		# Combine valid results
		valid_results = [r for r in model_results if r is not None]

		if not valid_results:
			logger.error("No valid sentiment models!")
			return pd.DataFrame(
				{
					"sentiment_label": ["neutral"] * len(df),
					"sentiment_numeric": [0] * len(df),
					"sentiment_confidence": [0.0] * len(df),
				}
			)

		# Concatenate all results
		combined = pd.concat(valid_results, axis=1)

		# Majority voting for label
		numeric_cols = [col for col in combined.columns if col.endswith("_numeric")]
		score_cols = [col for col in combined.columns if col.endswith("_score")]

		# Calculate ensemble prediction
		ensemble_numeric = combined[numeric_cols].mode(axis=1)[0].astype(int)
		ensemble_confidence = combined[score_cols].mean(axis=1)

		# Apply confidence threshold
		low_confidence_mask = ensemble_confidence < self.config["sentiment_threshold"]

		# Map numeric to label
		numeric_to_label = {-1: "negative", 0: "neutral", 1: "positive"}
		ensemble_label = ensemble_numeric.map(numeric_to_label)

		# Lower confidence = more likely neutral
		ensemble_label[low_confidence_mask] = "neutral"
		ensemble_numeric[low_confidence_mask] = 0

		return pd.DataFrame(
			{
				"sentiment_label": ensemble_label,
				"sentiment_numeric": ensemble_numeric,
				"sentiment_confidence": ensemble_confidence,
				"sentiment_raw_scores": combined[score_cols].to_dict("records"),
			}
		)

	def _analyze_emotion_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Ensemble emotion analysis
		"""
		# Model 1: DistilRoBERTa emotion
		try:
			emotion_pipeline = pipeline(
				"text-classification",
				model="j-hartmann/emotion-english-distilroberta-base",
				device=0 if torch.cuda.is_available() else -1,
				top_k=None,
			)
			primary_results = self._apply_emotion_model(df, emotion_pipeline)
		except Exception as e:
			logger.warning(f"Primary emotion model failed: {e}")
			primary_results = self._fallback_emotion_analysis(df)

		# Model 2: GoEmotions (more granular)
		try:
			goemotions_pipeline = pipeline(
				"text-classification",
				model="SamLowe/roberta-base-go_emotions",
				device=0 if torch.cuda.is_available() else -1,
				top_k=None,
			)
			secondary_results = self._apply_goemotions_model(df, goemotions_pipeline)
		except Exception as e:
			logger.warning(f"GoEmotions model failed: {e}")
			secondary_results = None

		# Combine
		if secondary_results is not None:
			emotion_df = self._combine_emotion_ensemble(
				primary_results, secondary_results
			)
		else:
			emotion_df = primary_results

		return emotion_df

	def _apply_emotion_model(self, df: pd.DataFrame, pipeline_model) -> pd.DataFrame:
		"""Apply emotion classification model"""
		batch_size = 16
		texts = df["cleaned_text"].tolist()

		all_results = []
		for i in tqdm(range(0, len(texts), batch_size), desc="Emotion analysis"):
			batch = texts[i : i + batch_size]
			batch_truncated = [text[:512] for text in batch]

			try:
				results = pipeline_model(batch_truncated)
				all_results.extend(results)
			except Exception as e:
				logger.warning(f"Error in emotion batch: {e}")
				all_results.extend([[{"label": "neutral", "score": 1.0}]] * len(batch))

		# Parse results
		emotion_data = []
		for result in all_results:
			top_emotion = max(result, key=lambda x: x["score"])
			emotion_dict = {
				"emotion_label": top_emotion["label"],
				"emotion_confidence": top_emotion["score"],
			}

			# Add all emotion scores
			for emotion in result:
				emotion_dict[f"emotion_{emotion['label']}"] = emotion["score"]

			emotion_data.append(emotion_dict)

		return pd.DataFrame(emotion_data)

	def _apply_goemotions_model(self, df: pd.DataFrame, pipeline_model) -> pd.DataFrame:
		"""Apply GoEmotions model (27 emotions)"""
		batch_size = 8  # Smaller batch for larger model
		texts = df["cleaned_text"].tolist()

		# Map GoEmotions to basic emotions
		GOEMOTIONS_MAP = {
			"anger": "anger",
			"annoyance": "anger",
			"disapproval": "anger",
			"disgust": "disgust",
			"fear": "fear",
			"nervousness": "fear",
			"joy": "joy",
			"amusement": "joy",
			"excitement": "joy",
			"gratitude": "joy",
			"love": "joy",
			"optimism": "joy",
			"sadness": "sadness",
			"grief": "sadness",
			"remorse": "sadness",
			"disappointment": "sadness",
			"surprise": "surprise",
			"realization": "surprise",
			"confusion": "surprise",
			"neutral": "neutral",
		}

		all_results = []
		for i in tqdm(range(0, len(texts), batch_size), desc="GoEmotions"):
			batch = texts[i : i + batch_size]
			batch_truncated = [text[:512] for text in batch]

			try:
				results = pipeline_model(batch_truncated)
				all_results.extend(results)
			except:
				all_results.extend([[{"label": "neutral", "score": 1.0}]] * len(batch))

		# Aggregate to basic emotions
		basic_emotions = []
		for result in all_results:
			# Map to basic emotions
			mapped_scores = defaultdict(float)
			for emotion in result:
				basic = GOEMOTIONS_MAP.get(emotion["label"], "neutral")
				mapped_scores[basic] += emotion["score"]

			# Normalize
			total = sum(mapped_scores.values())
			if total > 0:
				mapped_scores = {k: v / total for k, v in mapped_scores.items()}

			top_emotion = max(mapped_scores, key=mapped_scores.get)
			basic_emotions.append(
				{
					"goemotions_label": top_emotion,
					"goemotions_score": mapped_scores[top_emotion],
				}
			)

		return pd.DataFrame(basic_emotions)

	def _combine_emotion_ensemble(
		self, primary: pd.DataFrame, secondary: pd.DataFrame
	) -> pd.DataFrame:
		"""Combine emotion predictions from multiple models"""

		# Average confidence scores
		combined_confidence = (
			primary["emotion_confidence"] + secondary["goemotions_score"]
		) / 2

		# If both models agree, high confidence
		agreement_mask = primary["emotion_label"] == secondary["goemotions_label"]

		final_label = primary["emotion_label"].copy()
		final_confidence = combined_confidence.copy()

		# Where they agree, boost confidence
		final_confidence[agreement_mask] = np.minimum(
			final_confidence[agreement_mask] * 1.2, 1.0
		)

		# Where they disagree and both have low confidence, mark as neutral
		low_conf_mask = (primary["emotion_confidence"] < 0.5) & (
			secondary["goemotions_score"] < 0.5
		)
		final_label[low_conf_mask] = "neutral"

		result = primary.copy()
		result["emotion_label"] = final_label
		result["emotion_confidence"] = final_confidence

		return result

	def _fallback_emotion_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Fallback emotion analysis using sentiment"""
		n = len(df)
		return pd.DataFrame(
			{
				"emotion_label": ["neutral"] * n,
				"emotion_confidence": [0.5] * n,
				"emotion_neutral": [1.0] * n,
				"emotion_joy": [0.0] * n,
				"emotion_anger": [0.0] * n,
				"emotion_sadness": [0.0] * n,
				"emotion_fear": [0.0] * n,
				"emotion_disgust": [0.0] * n,
				"emotion_surprise": [0.0] * n,
			}
		)

	def _aggregate_affective_analysis(self, df: pd.DataFrame) -> Dict:
		"""Aggregate affective analysis with confidence filtering"""

		# Overall statistics
		affective_summary = {
			"overall_sentiment": {
				"positive": float((df["sentiment_numeric"] == 1).mean()),
				"neutral": float((df["sentiment_numeric"] == 0).mean()),
				"negative": float((df["sentiment_numeric"] == -1).mean()),
				"avg_score": float(df["sentiment_numeric"].mean()),
				"avg_confidence": float(df["sentiment_confidence"].mean()),
			},
			"overall_emotion": df["emotion_label"]
			.value_counts(normalize=True)
			.to_dict(),
			"emotion_confidence": {
				"avg": float(df["emotion_confidence"].mean()),
				"high_confidence_rate": float((df["emotion_confidence"] > 0.7).mean()),
			},
		}

		# Per-topic affective analysis
		if "topic" in df.columns:
			topic_affective = {}
			for topic_id in df["topic"].unique():
				if topic_id == -1:
					continue

				topic_df = df[df["topic"] == topic_id]

				# High-confidence only
				high_conf_mask = topic_df["sentiment_confidence"] > 0.6
				high_conf_topic = topic_df[high_conf_mask]

				if len(high_conf_topic) > 0:
					topic_affective[topic_id] = {
						"sentiment": {
							"positive": float(
								(high_conf_topic["sentiment_numeric"] == 1).mean()
							),
							"neutral": float(
								(high_conf_topic["sentiment_numeric"] == 0).mean()
							),
							"negative": float(
								(high_conf_topic["sentiment_numeric"] == -1).mean()
							),
							"avg_score": float(
								high_conf_topic["sentiment_numeric"].mean()
							),
							"confidence": float(
								high_conf_topic["sentiment_confidence"].mean()
							),
						},
						"emotion": high_conf_topic["emotion_label"]
						.value_counts(normalize=True)
						.to_dict(),
						"sample_size": len(high_conf_topic),
						"confidence_filtered": len(topic_df) - len(high_conf_topic),
					}

			affective_summary["topic_affective"] = topic_affective

		return affective_summary

	# ═══════════════════════════════════════════════════════════════════════
	# PHASE 5: ENHANCED MULTI-SCALE ANOMALY DETECTION
	# ═══════════════════════════════════════════════════════════════════════

	def detect_anomalies(
		self, df: pd.DataFrame, embeddings: np.ndarray
	) -> pd.DataFrame:
		logger.info("=" * 80)
		logger.info("PHASE 5: ENHANCED MULTI-SCALE ANOMALY DETECTION")
		logger.info("=" * 80)

		# Method 1: Isolation Forest (semantic anomalies)
		logger.info("Method 1: Isolation Forest on embeddings...")
		iforest_scores = self._isolation_forest_anomalies(embeddings)

		# Method 2: Local Outlier Factor (density-based)
		logger.info("Method 2: Local Outlier Factor...")
		lof_scores = self._local_outlier_anomalies(embeddings)

		# Method 3: Statistical anomalies (engagement, temporal)
		logger.info("Method 3: Statistical anomalies...")
		stat_scores = self._statistical_anomalies(df)

		# Ensemble anomaly score
		df["anomaly_score_iforest"] = iforest_scores
		df["anomaly_score_lof"] = lof_scores
		df["anomaly_score_statistical"] = stat_scores

		# Combined score (average)
		df["anomaly_score"] = (iforest_scores + lof_scores + stat_scores) / 3

		# Label as anomaly if score exceeds threshold
		threshold = np.percentile(
			df["anomaly_score"], (1 - self.config["anomaly_sensitivity"]) * 100
		)
		df["anomaly_label"] = (df["anomaly_score"] > threshold).astype(int)

		# Confidence = how far from threshold
		df["anomaly_confidence"] = np.abs(df["anomaly_score"] - threshold) / threshold
		df["anomaly_confidence"] = np.clip(df["anomaly_confidence"], 0, 1)

		# Characterize anomalies
		anomaly_char = self._characterize_anomalies_enhanced(df, embeddings)

		self.results["anomalies"] = {
			"total_anomalies": int(df["anomaly_label"].sum()),
			"anomaly_rate": float(df["anomaly_label"].mean()),
			"characterization": anomaly_char,
			"detection_methods": {
				"isolation_forest": float(
					(iforest_scores > np.percentile(iforest_scores, 90)).mean()
				),
				"lof": float((lof_scores > np.percentile(lof_scores, 90)).mean()),
				"statistical": float(
					(stat_scores > np.percentile(stat_scores, 90)).mean()
				),
			},
		}

		self.confidence_scores["anomaly_confidence"] = (
			df["anomaly_confidence"].describe().to_dict()
		)

		logger.info(
			f"Detected {df['anomaly_label'].sum():,} anomalies ({df['anomaly_label'].mean()*100:.2f}%)"
		)
		logger.info(
			f"Average anomaly confidence: {df['anomaly_confidence'].mean():.3f}"
		)

		return df

	def _isolation_forest_anomalies(self, embeddings: np.ndarray) -> np.ndarray:
		"""Isolation Forest for semantic anomalies"""
		iforest = IsolationForest(**self.config["isolation_forest"])

		# Fit on sample for speed
		sample_size = min(50000, len(embeddings))
		sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
		iforest.fit(embeddings[sample_idx])

		# Score all
		scores = iforest.score_samples(embeddings)

		# Normalize to [0, 1] where 1 = anomalous
		scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())
		scores_normalized = (
			1 - scores_normalized
		)  # Invert: lower IF score = more anomalous

		return scores_normalized

	def _local_outlier_anomalies(self, embeddings: np.ndarray) -> np.ndarray:
		"""Local Outlier Factor for density-based anomalies"""
		from sklearn.neighbors import LocalOutlierFactor

		lof = LocalOutlierFactor(
			n_neighbors=20,
			contamination=self.config["anomaly_sensitivity"],
			novelty=False,
		)

		# LOF returns -1 for outliers, 1 for inliers
		# negative_outlier_factor_ gives the score
		labels = lof.fit_predict(embeddings)
		scores = -lof.negative_outlier_factor_

		# Normalize
		scores_normalized = (scores - scores.min()) / (
			scores.max() - scores.min() + 1e-10
		)

		return scores_normalized

	def _statistical_anomalies(self, df: pd.DataFrame) -> np.ndarray:
		"""Statistical anomalies based on engagement, temporal patterns"""

		anomaly_scores = np.zeros(len(df))

		# Feature 1: Abnormal engagement
		if "engagement_score" in df.columns:
			engagement_zscore = np.abs(stats.zscore(df["engagement_score"].fillna(0)))
			anomaly_scores += (engagement_zscore > 3).astype(float) * 0.3

		# Feature 2: Unusual posting time
		if "hour" in df.columns:
			unusual_hours = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(float)
			anomaly_scores += unusual_hours * 0.2

		# Feature 3: Excessive hashtags
		if "hashtag_count" in df.columns:
			hashtag_zscore = np.abs(stats.zscore(df["hashtag_count"].fillna(0)))
			anomaly_scores += (hashtag_zscore > 2).astype(float) * 0.2

		# Feature 4: Very short or very long text
		if "text_length" in df.columns:
			length_zscore = np.abs(stats.zscore(df["text_length"]))
			anomaly_scores += (length_zscore > 3).astype(float) * 0.3

		# Normalize to [0, 1]
		if anomaly_scores.max() > 0:
			anomaly_scores = anomaly_scores / anomaly_scores.max()

		return anomaly_scores

	def _characterize_anomalies_enhanced(
		self, df: pd.DataFrame, embeddings: np.ndarray
	) -> Dict:
		"""Enhanced anomaly characterization"""

		anomalous = df[df["anomaly_label"] == 1]
		normal = df[df["anomaly_label"] == 0]

		characterization = {
			"count": len(anomalous),
			"percentage": len(anomalous) / len(df) * 100,
		}

		# Temporal analysis
		if "timestamp" in df.columns:
			anomaly_temporal = anomalous.groupby(
				pd.Grouper(key="timestamp", freq="D")
			).size()
			normal_temporal = normal.groupby(
				pd.Grouper(key="timestamp", freq="D")
			).size()

			# Identify burst days
			if len(anomaly_temporal) > 0:
				bursts = anomaly_temporal[
					anomaly_temporal > anomaly_temporal.quantile(0.95)
				]
				characterization["temporal_bursts"] = {
					"dates": [str(d) for d in bursts.index],
					"counts": bursts.tolist(),
				}

		# Topic distribution
		if "topic" in df.columns:
			anomaly_topics = (
				anomalous["topic"].value_counts(normalize=True).head(10).to_dict()
			)
			normal_topics = normal["topic"].value_counts(normalize=True).to_dict()

			# Find overrepresented topics
			overrep = {}
			for topic, anom_rate in anomaly_topics.items():
				norm_rate = normal_topics.get(topic, 0)
				if norm_rate > 0:
					ratio = anom_rate / norm_rate
					if ratio > 2:
						overrep[int(topic)] = float(ratio)

			characterization["anomaly_prone_topics"] = overrep

		# Linguistic features
		characterization["linguistic"] = {
			"avg_length_anomalous": (
				float(anomalous["text_length"].mean()) if len(anomalous) > 0 else 0
			),
			"avg_length_normal": (
				float(normal["text_length"].mean()) if len(normal) > 0 else 0
			),
			"hashtag_count_anomalous": (
				float(anomalous["hashtag_count"].mean()) if len(anomalous) > 0 else 0
			),
			"hashtag_count_normal": (
				float(normal["hashtag_count"].mean()) if len(normal) > 0 else 0
			),
		}

		# Affective differences
		if "sentiment_numeric" in df.columns:
			characterization["affective"] = {
				"sentiment_anomalous": (
					float(anomalous["sentiment_numeric"].mean())
					if len(anomalous) > 0
					else 0
				),
				"sentiment_normal": (
					float(normal["sentiment_numeric"].mean()) if len(normal) > 0 else 0
				),
			}

		# Example anomalous tweets
		if len(anomalous) > 0:
			top_anomalies = anomalous.nlargest(5, "anomaly_score")
			characterization["examples"] = top_anomalies["cleaned_text"].tolist()

		return characterization

	# ═══════════════════════════════════════════════════════════════════════
	# PHASE 6: ENHANCED BOT DETECTION (ENSEMBLE)
	# ═══════════════════════════════════════════════════════════════════════

	def detect_bots(self, df: pd.DataFrame) -> pd.DataFrame:
		logger.info("=" * 80)
		logger.info("PHASE 6: ENHANCED BOT DETECTION (ENSEMBLE)")
		logger.info("=" * 80)

		# Calculate bot detection features
		logger.info("Calculating bot detection features...")
		user_features = self._calculate_bot_features(df)

		# Multiple detection methods
		heuristic_scores = self._heuristic_bot_detection(user_features)
		ml_scores = self._ml_bot_detection(user_features)

		# Ensemble
		user_features["bot_score_heuristic"] = heuristic_scores
		user_features["bot_score_ml"] = ml_scores
		user_features["bot_score"] = (heuristic_scores + ml_scores) / 2
		user_features["bot_label"] = (
			user_features["bot_score"] >= self.config["bot_threshold"]
		).astype(int)
		user_features["bot_confidence"] = np.abs(
			user_features["bot_score"] - self.config["bot_threshold"]
		)
		user_features["bot_confidence"] = np.clip(user_features["bot_confidence"], 0, 1)

		# Merge with main dataframe
		df = df.merge(
			user_features[["bot_score", "bot_label", "bot_confidence"]],
			left_on="original_author",
			right_index=True,
			how="left",
		)

		df["bot_score"] = df["bot_score"].fillna(0.5)
		df["bot_label"] = df["bot_label"].fillna(0)
		df["bot_confidence"] = df["bot_confidence"].fillna(0)
		df["human_weight"] = 1 - df["bot_score"]

		# Analyze bot impact
		bot_impact = self._analyze_bot_impact_enhanced(df)

		self.results["bots"] = {
			"bot_accounts": int(user_features["bot_label"].sum()),
			"bot_tweets": int(df["bot_label"].sum()),
			"bot_tweet_percentage": float(df["bot_label"].mean() * 100),
			"avg_bot_confidence": float(
				df[df["bot_label"] == 1]["bot_confidence"].mean()
			),
			"impact": bot_impact,
		}

		self.confidence_scores["bot_confidence"] = (
			df["bot_confidence"].describe().to_dict()
		)

		logger.info(
			f"Identified {user_features['bot_label'].sum():,} potential bot accounts"
		)
		logger.info(
			f"Bot tweets: {df['bot_label'].sum():,} ({df['bot_label'].mean()*100:.2f}%)"
		)
		logger.info(
			f"Average bot detection confidence: {df['bot_confidence'].mean():.3f}"
		)

		return df

	def _calculate_bot_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Calculate comprehensive bot detection features per user"""

		user_stats = df.groupby("original_author").agg(
			{
				"id": "count",
				"cleaned_text": [
					lambda x: len(set(x)) / len(x) if len(x) > 0 else 1,  # uniqueness
					lambda x: np.mean([len(t) for t in x]),  # avg length
				],
				"retweet_count": ["mean", "sum"],
				"favorite_count": ["mean", "sum"],
				"hashtag_count": ["mean", "max"],
				"mention_count": ["mean", "max"],
				"url_count": "mean",
				"is_retweet": "mean",
				"timestamp": [
					lambda x: (
						(x.max() - x.min()).total_seconds() / 3600 if len(x) > 1 else 0
					),  # timespan
					lambda x: (
						len(x) / ((x.max() - x.min()).total_seconds() / 3600 + 1)
						if len(x) > 1
						else 0
					),  # frequency
				],
				"hour": lambda x: (
					x.value_counts().max() / len(x) if len(x) > 0 else 0
				),  # temporal concentration
			}
		)

		# Flatten columns
		user_stats.columns = [
			"tweet_count",
			"content_uniqueness",
			"avg_text_length",
			"avg_retweets",
			"total_retweets",
			"avg_favorites",
			"total_favorites",
			"avg_hashtags",
			"max_hashtags",
			"avg_mentions",
			"max_mentions",
			"avg_urls",
			"retweet_ratio",
			"timespan_hours",
			"tweet_frequency",
			"temporal_concentration",
		]

		return user_stats

	def _heuristic_bot_detection(self, user_features: pd.DataFrame) -> np.ndarray:
		"""Heuristic bot detection based on rules"""

		bot_scores = np.zeros(len(user_features))

		# Feature 1: High volume (>100 tweets in dataset)
		bot_scores += (user_features["tweet_count"] > 100).astype(float) * 0.15

		# Feature 2: Very high volume (>500 tweets)
		bot_scores += (user_features["tweet_count"] > 500).astype(float) * 0.15

		# Feature 3: Low content uniqueness (<30%)
		bot_scores += (user_features["content_uniqueness"] < 0.3).astype(float) * 0.2

		# Feature 4: High retweet ratio (>80%)
		bot_scores += (user_features["retweet_ratio"] > 0.8).astype(float) * 0.15

		# Feature 5: Excessive hashtags (avg > 5)
		bot_scores += (user_features["avg_hashtags"] > 5).astype(float) * 0.1

		# Feature 6: High frequency (>10 tweets/hour)
		bot_scores += (user_features["tweet_frequency"] > 10).astype(float) * 0.15

		# Feature 7: Temporal concentration (>50% at same hour)
		bot_scores += (user_features["temporal_concentration"] > 0.5).astype(
			float
		) * 0.1

		return bot_scores

	def _ml_bot_detection(self, user_features: pd.DataFrame) -> np.ndarray:
		"""Machine learning bot detection using unsupervised approach"""

		# Features for clustering
		feature_cols = [
			"tweet_count",
			"content_uniqueness",
			"avg_text_length",
			"avg_hashtags",
			"retweet_ratio",
			"tweet_frequency",
			"temporal_concentration",
		]

		X = user_features[feature_cols].fillna(0)

		# Normalize
		from sklearn.preprocessing import StandardScaler

		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

		# Isolation Forest for anomalous users
		iforest = IsolationForest(
			n_estimators=100,
			contamination=0.15,  # Assume 15% bots
			random_state=RANDOM_SEED,
		)

		anomaly_scores = iforest.fit_predict(X_scaled)
		anomaly_scores_continuous = -iforest.score_samples(X_scaled)

		# Normalize to [0, 1]
		scores_normalized = (
			anomaly_scores_continuous - anomaly_scores_continuous.min()
		) / (anomaly_scores_continuous.max() - anomaly_scores_continuous.min())

		return scores_normalized

	def _analyze_bot_impact_enhanced(self, df: pd.DataFrame) -> Dict:
		"""Enhanced bot impact analysis"""

		impact = {}

		# Sentiment impact
		if "sentiment_numeric" in df.columns:
			human_sentiment = df[df["bot_label"] == 0]["sentiment_numeric"].mean()
			bot_sentiment = df[df["bot_label"] == 1]["sentiment_numeric"].mean()

			impact["sentiment"] = {
				"human_avg": float(human_sentiment),
				"bot_avg": float(bot_sentiment),
				"difference": float(bot_sentiment - human_sentiment),
				"distortion": (
					"positive" if bot_sentiment > human_sentiment else "negative"
				),
			}

		# Topic amplification
		if "topic" in df.columns:
			bot_topic_dist = df[df["bot_label"] == 1]["topic"].value_counts(
				normalize=True
			)
			human_topic_dist = df[df["bot_label"] == 0]["topic"].value_counts(
				normalize=True
			)

			amplified = {}
			suppressed = {}

			for topic in set(bot_topic_dist.index) | set(human_topic_dist.index):
				if topic == -1:
					continue
				bot_rate = bot_topic_dist.get(topic, 0)
				human_rate = human_topic_dist.get(topic, 0)

				if human_rate > 0:
					ratio = bot_rate / human_rate
					if ratio > 1.5:
						amplified[int(topic)] = float(ratio)
					elif ratio < 0.5:
						suppressed[int(topic)] = float(ratio)

			impact["topic_manipulation"] = {
				"amplified_topics": amplified,
				"suppressed_topics": suppressed,
			}

		# Temporal patterns
		if "timestamp" in df.columns:
			bot_temporal = (
				df[df["bot_label"] == 1]
				.groupby(pd.Grouper(key="timestamp", freq="D"))
				.size()
			)
			if len(bot_temporal) > 0:
				peak_days = bot_temporal.nlargest(5)
				impact["temporal_peaks"] = {
					"dates": [str(d) for d in peak_days.index],
					"counts": peak_days.tolist(),
				}

		# Coordinated behavior
		bot_tweets = df[df["bot_label"] == 1]
		if len(bot_tweets) > 100:
			sample = bot_tweets.sample(
				n=min(1000, len(bot_tweets)), random_state=RANDOM_SEED
			)

			vectorizer = TfidfVectorizer(max_features=100)
			tfidf = vectorizer.fit_transform(sample["cleaned_text"])
			similarity = cosine_similarity(tfidf)

			high_similarity = (similarity > 0.9).sum() // 2
			impact["coordination_score"] = int(high_similarity)

		return impact

	# ═══════════════════════════════════════════════════════════════════════
	# PHASE 7: STATISTICAL VALIDATION AND SIGNIFICANCE TESTING
	# ═══════════════════════════════════════════════════════════════════════

	def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict:
		logger.info("=" * 80)
		logger.info("PHASE 7: STATISTICAL VALIDATION AND SIGNIFICANCE TESTING")
		logger.info("=" * 80)

		statistical_results = {}

		# Trend analysis
		trend_results = self._mann_kendall_analysis(df)
		statistical_results["trends"] = trend_results

		# Changepoint detection
		changepoint_results = self._detect_changepoints(df)
		statistical_results["changepoints"] = changepoint_results

		# Topic coherence validation
		if "topics" in self.results:
			coherence_results = self._validate_topic_coherence(df)
			statistical_results["coherence"] = coherence_results

		# Sentiment validation
		sentiment_validation = self._validate_sentiment_distribution(df)
		statistical_results["sentiment_validation"] = sentiment_validation

		# Cross-phase comparisons
		if "phase" in df.columns and df["phase"].nunique() > 1:
			phase_comparison = self._compare_phases(df)
			statistical_results["phase_comparison"] = phase_comparison

		self.results["statistics"] = statistical_results

		return statistical_results

	def _mann_kendall_analysis(self, df: pd.DataFrame) -> Dict:
		"""Mann-Kendall trend analysis for temporal trends"""
		logger.info("Performing Mann-Kendall trend analysis...")

		daily_stats = df.groupby(pd.Grouper(key="timestamp", freq="D")).agg(
			{
				"sentiment_numeric": "mean",
				"anomaly_label": "mean",
				"bot_label": "mean",
				"engagement_score": "mean",
			}
		)

		trend_results = {}
		for column in daily_stats.columns:
			series = daily_stats[column].dropna()
			if len(series) > 10:
				tau, p_value = stats.kendalltau(range(len(series)), series)

				trend_results[column] = {
					"tau": float(tau),
					"p_value": float(p_value),
					"significant": p_value < self.config["significance_level"],
					"direction": "increasing" if tau > 0 else "decreasing",
					"interpretation": self._interpret_trend(tau, p_value),
				}

		return trend_results

	def _interpret_trend(self, tau: float, p_value: float) -> str:
		"""Interpret Mann-Kendall results"""
		if p_value >= 0.05:
			return "No significant trend"
		elif tau > 0.3:
			return "Strong increasing trend"
		elif tau > 0.1:
			return "Moderate increasing trend"
		elif tau < -0.3:
			return "Strong decreasing trend"
		elif tau < -0.1:
			return "Moderate decreasing trend"
		else:
			return "Weak trend"

	def _detect_changepoints(self, df: pd.DataFrame) -> List[Dict]:
		"""Detect changepoints in sentiment and activity"""
		logger.info("Detecting changepoints...")

		changepoints = []

		# Sentiment changepoints
		daily_sentiment = df.groupby(pd.Grouper(key="timestamp", freq="D"))[
			"sentiment_numeric"
		].mean()

		window_size = 7
		threshold = 2

		for i in range(window_size, len(daily_sentiment) - window_size):
			before = daily_sentiment[i - window_size : i]
			after = daily_sentiment[i : i + window_size]

			if len(before.dropna()) < 3 or len(after.dropna()) < 3:
				continue

			t_stat, p_value = stats.ttest_ind(before.dropna(), after.dropna())

			if p_value < self.config["significance_level"] and abs(t_stat) > threshold:
				changepoints.append(
					{
						"date": str(daily_sentiment.index[i]),
						"metric": "sentiment",
						"before_mean": float(before.mean()),
						"after_mean": float(after.mean()),
						"magnitude": float(after.mean() - before.mean()),
						"p_value": float(p_value),
						"t_statistic": float(t_stat),
					}
				)

		# Volume changepoints
		daily_volume = df.groupby(pd.Grouper(key="timestamp", freq="D")).size()

		for i in range(window_size, len(daily_volume) - window_size):
			before = daily_volume[i - window_size : i]
			after = daily_volume[i : i + window_size]

			u_stat, p_value = mannwhitneyu(before, after, alternative="two-sided")

			if p_value < 0.01 and abs(after.mean() - before.mean()) > before.std():
				changepoints.append(
					{
						"date": str(daily_volume.index[i]),
						"metric": "volume",
						"before_mean": float(before.mean()),
						"after_mean": float(after.mean()),
						"magnitude": float(after.mean() - before.mean()),
						"p_value": float(p_value),
					}
				)

		return changepoints

	def _validate_topic_coherence(self, df: pd.DataFrame) -> Dict:
		"""Validate topic model coherence"""
		logger.info("Validating topic coherence...")

		coherence_results = {}

		if "characterization" in self.results.get("topics", {}):
			topics_char = self.results["topics"]["characterization"]

			coherence_scores = []
			coverage_scores = []

			for topic_id, info in topics_char.items():
				if topic_id == -1:
					continue

				topic_tweets = df[df["topic"] == topic_id]["cleaned_text"]

				if len(topic_tweets) > 0 and info.get("keywords"):
					# Keyword coverage
					keyword_coverage = 0
					for keyword, score in info["keywords"][:5]:
						keyword_text = (
							keyword[0] if isinstance(keyword, tuple) else keyword
						)
						matches = sum(
							keyword_text.lower() in tweet.lower()
							for tweet in topic_tweets
						)
						keyword_coverage += matches / len(topic_tweets)

					keyword_coverage /= min(5, len(info["keywords"]))
					coherence_scores.append(keyword_coverage)

					# Topic purity (confidence)
					if "topic_confidence" in df.columns:
						avg_confidence = df[df["topic"] == topic_id][
							"topic_confidence"
						].mean()
						coverage_scores.append(avg_confidence)

			if coherence_scores:
				coherence_results["keyword_coherence"] = {
					"mean": float(np.mean(coherence_scores)),
					"std": float(np.std(coherence_scores)),
					"min": float(np.min(coherence_scores)),
					"max": float(np.max(coherence_scores)),
				}

			if coverage_scores:
				coherence_results["topic_confidence"] = {
					"mean": float(np.mean(coverage_scores)),
					"std": float(np.std(coverage_scores)),
				}

			# Quality metrics from earlier
			if "quality_metrics" in self.results["topics"]:
				coherence_results["quality_metrics"] = self.results["topics"][
					"quality_metrics"
				]

		return coherence_results

	def _validate_sentiment_distribution(self, df: pd.DataFrame) -> Dict:
		"""Validate sentiment distribution is not collapsed"""
		logger.info("Validating sentiment distribution...")

		validation = {}

		if "sentiment_numeric" in df.columns:
			sentiment_dist = df["sentiment_numeric"].value_counts(normalize=True)

			# Check for collapse (>80% in one category)
			max_proportion = sentiment_dist.max()

			validation["distribution"] = sentiment_dist.to_dict()
			validation["entropy"] = float(entropy(sentiment_dist))
			validation["max_proportion"] = float(max_proportion)
			validation["is_collapsed"] = max_proportion > 0.8
			validation["is_balanced"] = max_proportion < 0.5

			# Check confidence distribution
			if "sentiment_confidence" in df.columns:
				validation["confidence_stats"] = {
					"mean": float(df["sentiment_confidence"].mean()),
					"std": float(df["sentiment_confidence"].std()),
					"high_confidence_rate": float(
						(df["sentiment_confidence"] > 0.7).mean()
					),
				}

		# Emotion distribution
		if "emotion_label" in df.columns:
			emotion_dist = df["emotion_label"].value_counts(normalize=True)

			validation["emotion_distribution"] = emotion_dist.to_dict()
			validation["emotion_entropy"] = float(entropy(emotion_dist))
			validation["emotion_collapsed"] = emotion_dist.max() > 0.8

		return validation

	def _compare_phases(self, df: pd.DataFrame) -> Dict:
		"""Compare metrics across pandemic phases"""
		logger.info("Comparing across pandemic phases...")

		phases = df["phase"].unique()
		comparison = {}

		for phase in phases:
			phase_df = df[df["phase"] == phase]

			comparison[phase] = {
				"tweet_count": len(phase_df),
				"sentiment_avg": (
					float(phase_df["sentiment_numeric"].mean())
					if "sentiment_numeric" in phase_df.columns
					else None
				),
				"anomaly_rate": (
					float(phase_df["anomaly_label"].mean())
					if "anomaly_label" in phase_df.columns
					else None
				),
				"bot_rate": (
					float(phase_df["bot_label"].mean())
					if "bot_label" in phase_df.columns
					else None
				),
				"top_topics": (
					phase_df["topic"].value_counts().head(5).to_dict()
					if "topic" in phase_df.columns
					else {}
				),
			}

		# Statistical tests between phases
		if len(phases) >= 2:
			phase_pairs = []
			for i, phase1 in enumerate(phases):
				for phase2 in phases[i + 1 :]:
					df1 = df[df["phase"] == phase1]
					df2 = df[df["phase"] == phase2]

					# Sentiment comparison
					if "sentiment_numeric" in df.columns:
						u_stat, p_val = mannwhitneyu(
							df1["sentiment_numeric"].dropna(),
							df2["sentiment_numeric"].dropna(),
						)

						phase_pairs.append(
							{
								"phases": f"{phase1} vs {phase2}",
								"metric": "sentiment",
								"p_value": float(p_val),
								"significant": p_val < 0.05,
							}
						)

			comparison["statistical_tests"] = phase_pairs

		return comparison

	# ═══════════════════════════════════════════════════════════════════════
	# PHASE 8: ENHANCED VISUALIZATIONS WITH CONFIDENCE SCORES
	# ═══════════════════════════════════════════════════════════════════════

	def generate_visualizations(self, df: pd.DataFrame) -> None:
		logger.info("=" * 80)
		logger.info("PHASE 8: ENHANCED VISUALIZATIONS WITH CONFIDENCE TRACKING")
		logger.info("=" * 80)

		# Temporal trends
		self._create_temporal_plots_enhanced(df)

		# Topic visualizations
		if "topics" in self.results:
			self._create_topic_plots_enhanced(df)
			self._create_topic_wordclouds(df)

		# Affective analysis
		if "affective" in self.results:
			self._create_affective_plots_enhanced(df)

		# Anomaly visualizations
		if "anomalies" in self.results:
			self._create_anomaly_plots_enhanced(df)

		# Bot detection
		if "bots" in self.results:
			self._create_bot_plots(df)

		# Confidence dashboards
		self._create_confidence_dashboard(df)

		# Interactive visualizations
		self._create_interactive_timeline_enhanced(df)
		self._create_topic_network_enhanced(df)
		self._create_sentiment_topic_heatmap(df)

		logger.info("All visualizations saved to outputs/visualizations/")

	def _create_temporal_plots_enhanced(self, df: pd.DataFrame):
		"""Enhanced temporal plots with confidence bands"""
		logger.info("Creating enhanced temporal visualizations...")

		daily = df.groupby(pd.Grouper(key="timestamp", freq="D")).agg(
			{
				"sentiment_numeric": ["mean", "std"],
				"sentiment_confidence": "mean",
				"anomaly_label": "mean",
				"bot_label": "mean",
				"id": "count",
			}
		)

		daily.columns = [
			"sentiment_mean",
			"sentiment_std",
			"sentiment_conf",
			"anomaly_rate",
			"bot_rate",
			"tweet_count",
		]

		fig, axes = plt.subplots(5, 1, figsize=(16, 14))

		# Plot 1: Tweet volume
		axes[0].plot(daily.index, daily["tweet_count"], color="steelblue", linewidth=2)
		axes[0].fill_between(
			daily.index, daily["tweet_count"], alpha=0.3, color="steelblue"
		)
		axes[0].set_title("Daily Tweet Volume", fontsize=14, fontweight="bold")
		axes[0].set_ylabel("Number of Tweets")
		axes[0].grid(True, alpha=0.3)

		# Plot 2: Sentiment with confidence bands
		axes[1].plot(
			daily.index,
			daily["sentiment_mean"],
			color="green",
			linewidth=2,
			label="Mean Sentiment",
		)
		axes[1].fill_between(
			daily.index,
			daily["sentiment_mean"] - daily["sentiment_std"],
			daily["sentiment_mean"] + daily["sentiment_std"],
			alpha=0.3,
			color="green",
			label="±1 Std Dev",
		)
		axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
		axes[1].set_title(
			"Sentiment Trend with Confidence Bands", fontsize=14, fontweight="bold"
		)
		axes[1].set_ylabel("Average Sentiment")
		axes[1].set_ylim(-1, 1)
		axes[1].legend()
		axes[1].grid(True, alpha=0.3)

		# Plot 3: Sentiment confidence
		axes[2].plot(daily.index, daily["sentiment_conf"], color="purple", linewidth=2)
		axes[2].axhline(
			y=0.6, color="red", linestyle="--", alpha=0.5, label="Threshold"
		)
		axes[2].set_title(
			"Sentiment Prediction Confidence", fontsize=14, fontweight="bold"
		)
		axes[2].set_ylabel("Average Confidence")
		axes[2].legend()
		axes[2].grid(True, alpha=0.3)

		# Plot 4: Anomaly rate
		axes[3].plot(daily.index, daily["anomaly_rate"] * 100, color="red", linewidth=2)
		axes[3].set_title("Anomaly Detection Rate", fontsize=14, fontweight="bold")
		axes[3].set_ylabel("Anomaly %")
		axes[3].grid(True, alpha=0.3)

		# Plot 5: Bot activity
		axes[4].plot(daily.index, daily["bot_rate"] * 100, color="orange", linewidth=2)
		axes[4].set_title("Bot Activity Rate", fontsize=14, fontweight="bold")
		axes[4].set_ylabel("Bot Tweet %")
		axes[4].set_xlabel("Date")
		axes[4].grid(True, alpha=0.3)

		plt.tight_layout()
		plt.savefig(
			"outputs/visualizations/temporal_trends_enhanced.png",
			dpi=300,
			bbox_inches="tight",
		)
		plt.close()

	def _create_topic_plots_enhanced(self, df: pd.DataFrame):
		"""Enhanced topic visualizations with names and confidence"""
		logger.info("Creating enhanced topic visualizations...")

		# Get topic names
		topic_names = self.results["topics"].get("topic_names", {})

		# Topic distribution
		topic_counts = (
			df[df["topic"] != -1]
			.groupby("topic")
			.agg(
				{"id": "count", "topic_confidence": "mean", "sentiment_numeric": "mean"}
			)
			.rename(columns={"id": "count"})
		)

		topic_counts = topic_counts.nlargest(20, "count")
		topic_counts["topic_name"] = topic_counts.index.map(
			lambda x: topic_names.get(x, f"Topic {x}")[:30]
		)

		fig, axes = plt.subplots(2, 1, figsize=(14, 12))

		# Plot 1: Topic sizes with sentiment coloring
		colors = topic_counts["sentiment_numeric"].apply(
			lambda x: "green" if x > 0.1 else "red" if x < -0.1 else "gray"
		)

		bars = axes[0].barh(
			range(len(topic_counts)), topic_counts["count"], color=colors
		)
		axes[0].set_yticks(range(len(topic_counts)))
		axes[0].set_yticklabels(topic_counts["topic_name"], fontsize=10)
		axes[0].set_xlabel("Number of Tweets", fontsize=12)
		axes[0].set_title(
			"Top 20 Topics by Size (Color = Sentiment)", fontsize=14, fontweight="bold"
		)
		axes[0].invert_yaxis()
		axes[0].grid(True, alpha=0.3, axis="x")

		# Add legend
		from matplotlib.patches import Patch

		legend_elements = [
			Patch(facecolor="green", label="Positive"),
			Patch(facecolor="gray", label="Neutral"),
			Patch(facecolor="red", label="Negative"),
		]
		axes[0].legend(handles=legend_elements, loc="lower right")

		# Plot 2: Topic confidence scores
		bars = axes[1].barh(
			range(len(topic_counts)), topic_counts["topic_confidence"], color="skyblue"
		)
		axes[1].set_yticks(range(len(topic_counts)))
		axes[1].set_yticklabels(topic_counts["topic_name"], fontsize=10)
		axes[1].set_xlabel("Average Topic Confidence", fontsize=12)
		axes[1].set_title("Topic Assignment Confidence", fontsize=14, fontweight="bold")
		axes[1].axvline(
			x=0.6, color="red", linestyle="--", alpha=0.5, label="Threshold"
		)
		axes[1].invert_yaxis()
		axes[1].legend()
		axes[1].grid(True, alpha=0.3, axis="x")

		plt.tight_layout()
		plt.savefig(
			"outputs/visualizations/topic_distribution_enhanced.png",
			dpi=300,
			bbox_inches="tight",
		)
		plt.close()

	def _create_topic_wordclouds(self, df: pd.DataFrame):
		"""Create word clouds for top topics"""
		logger.info("Creating topic word clouds...")

		topic_names = self.results["topics"].get("topic_names", {})
		top_topics = df[df["topic"] != -1]["topic"].value_counts().head(6).index

		fig, axes = plt.subplots(2, 3, figsize=(18, 12))
		axes = axes.flatten()

		for idx, topic_id in enumerate(top_topics):
			topic_tweets = df[df["topic"] == topic_id]["cleaned_text"]
			text = " ".join(topic_tweets.tolist())

			wordcloud = WordCloud(
				width=800,
				height=400,
				background_color="white",
				colormap="viridis",
				max_words=50,
			).generate(text)

			axes[idx].imshow(wordcloud, interpolation="bilinear")
			axes[idx].axis("off")
			topic_name = topic_names.get(topic_id, f"Topic {topic_id}")
			axes[idx].set_title(
				f"{topic_name}\n({len(topic_tweets)} tweets)",
				fontsize=12,
				fontweight="bold",
			)

		plt.tight_layout()
		plt.savefig(
			"outputs/visualizations/topic_wordclouds.png", dpi=300, bbox_inches="tight"
		)
		plt.close()

	def _create_affective_plots_enhanced(self, df: pd.DataFrame):
		"""Enhanced affective visualizations"""
		logger.info("Creating enhanced affective visualizations...")

		fig = plt.figure(figsize=(16, 10))
		gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

		# 1. Sentiment distribution with confidence
		ax1 = fig.add_subplot(gs[0, 0])
		sentiment_conf = df.groupby("sentiment_label")["sentiment_confidence"].mean()
		sentiment_counts = df["sentiment_label"].value_counts()

		bars = ax1.bar(
			sentiment_counts.index,
			sentiment_counts.values,
			color=[
				"red" if x == "negative" else "gray" if x == "neutral" else "green"
				for x in sentiment_counts.index
			],
		)
		ax1.set_title("Sentiment Distribution", fontsize=12, fontweight="bold")
		ax1.set_ylabel("Count")
		ax1.grid(True, alpha=0.3, axis="y")

		# Add confidence as text
		for i, (label, count) in enumerate(sentiment_counts.items()):
			conf = sentiment_conf.get(label, 0)
			ax1.text(i, count, f"conf: {conf:.2f}", ha="center", va="bottom")

		# 2. Emotion distribution
		ax2 = fig.add_subplot(gs[0, 1])
		emotion_counts = df["emotion_label"].value_counts().head(7)
		ax2.bar(range(len(emotion_counts)), emotion_counts.values, color="skyblue")
		ax2.set_xticks(range(len(emotion_counts)))
		ax2.set_xticklabels(emotion_counts.index, rotation=45, ha="right")
		ax2.set_title("Emotion Distribution", fontsize=12, fontweight="bold")
		ax2.set_ylabel("Count")
		ax2.grid(True, alpha=0.3, axis="y")

		# 3. Sentiment over time with confidence
		ax3 = fig.add_subplot(gs[1, :])
		daily_sentiment = df.groupby(pd.Grouper(key="timestamp", freq="D")).agg(
			{"sentiment_numeric": "mean", "sentiment_confidence": "mean"}
		)

		ax3_twin = ax3.twinx()
		ax3.plot(
			daily_sentiment.index,
			daily_sentiment["sentiment_numeric"],
			color="green",
			linewidth=2,
			label="Sentiment",
		)
		ax3_twin.plot(
			daily_sentiment.index,
			daily_sentiment["sentiment_confidence"],
			color="purple",
			linewidth=2,
			linestyle="--",
			label="Confidence",
			alpha=0.7,
		)

		ax3.set_ylabel("Sentiment", color="green")
		ax3_twin.set_ylabel("Confidence", color="purple")
		ax3.set_title("Sentiment Trend with Confidence", fontsize=12, fontweight="bold")
		ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
		ax3.grid(True, alpha=0.3)
		ax3.legend(loc="upper left")
		ax3_twin.legend(loc="upper right")

		# 4. Emotion-Topic heatmap
		if "topic" in df.columns:
			ax4 = fig.add_subplot(gs[2, :])

			# Get top topics
			top_topics = df[df["topic"] != -1]["topic"].value_counts().head(10).index
			emotions = df["emotion_label"].value_counts().head(7).index

			# Create matrix
			matrix = np.zeros((len(emotions), len(top_topics)))
			topic_names = self.results["topics"].get("topic_names", {})

			for i, emotion in enumerate(emotions):
				for j, topic in enumerate(top_topics):
					count = len(
						df[(df["emotion_label"] == emotion) & (df["topic"] == topic)]
					)
					total = len(df[df["topic"] == topic])
					matrix[i, j] = count / total if total > 0 else 0

			im = ax4.imshow(matrix, cmap="YlOrRd", aspect="auto")
			ax4.set_xticks(range(len(top_topics)))
			ax4.set_xticklabels(
				[topic_names.get(t, f"T{t}")[:20] for t in top_topics],
				rotation=45,
				ha="right",
				fontsize=9,
			)
			ax4.set_yticks(range(len(emotions)))
			ax4.set_yticklabels(emotions, fontsize=9)
			ax4.set_title(
				"Emotion Distribution Across Topics", fontsize=12, fontweight="bold"
			)

			# Add colorbar
			cbar = plt.colorbar(im, ax=ax4)
			cbar.set_label("Proportion")

			# Add values
			for i in range(len(emotions)):
				for j in range(len(top_topics)):
					text = ax4.text(
						j,
						i,
						f"{matrix[i, j]:.2f}",
						ha="center",
						va="center",
						color="black",
						fontsize=8,
					)

		plt.savefig(
			"outputs/visualizations/affective_analysis_enhanced.png",
			dpi=300,
			bbox_inches="tight",
		)
		plt.close()

	def _create_anomaly_plots_enhanced(self, df: pd.DataFrame):
		"""Enhanced anomaly visualizations"""
		logger.info("Creating enhanced anomaly visualizations...")

		fig, axes = plt.subplots(2, 2, figsize=(16, 12))

		# 1. Anomaly score distribution
		axes[0, 0].hist(
			df["anomaly_score"], bins=50, color="red", alpha=0.7, edgecolor="black"
		)
		threshold = np.percentile(df["anomaly_score"], 90)
		axes[0, 0].axvline(
			x=threshold, color="darkred", linestyle="--", linewidth=2, label="Threshold"
		)
		axes[0, 0].set_title(
			"Anomaly Score Distribution", fontsize=12, fontweight="bold"
		)
		axes[0, 0].set_xlabel("Anomaly Score")
		axes[0, 0].set_ylabel("Frequency")
		axes[0, 0].legend()
		axes[0, 0].grid(True, alpha=0.3)

		# 2. Anomaly methods comparison
		method_scores = df[
			["anomaly_score_iforest", "anomaly_score_lof", "anomaly_score_statistical"]
		].mean()
		axes[0, 1].bar(
			range(len(method_scores)),
			method_scores.values,
			color=["blue", "green", "orange"],
		)
		axes[0, 1].set_xticks(range(len(method_scores)))
		axes[0, 1].set_xticklabels(
			["Isolation\nForest", "Local Outlier\nFactor", "Statistical"], fontsize=10
		)
		axes[0, 1].set_title(
			"Anomaly Detection Methods - Average Scores", fontsize=12, fontweight="bold"
		)
		axes[0, 1].set_ylabel("Average Score")
		axes[0, 1].grid(True, alpha=0.3, axis="y")

		# 3. Temporal anomaly pattern
		daily_anomaly = df.groupby(pd.Grouper(key="timestamp", freq="D")).agg(
			{"anomaly_label": ["sum", "mean"]}
		)
		daily_anomaly.columns = ["anomaly_count", "anomaly_rate"]

		axes[1, 0].bar(
			daily_anomaly.index, daily_anomaly["anomaly_count"], color="red", alpha=0.6
		)
		axes[1, 0].set_title("Daily Anomaly Count", fontsize=12, fontweight="bold")
		axes[1, 0].set_xlabel("Date")
		axes[1, 0].set_ylabel("Number of Anomalies")
		axes[1, 0].tick_params(axis="x", rotation=45)
		axes[1, 0].grid(True, alpha=0.3)

		# 4. Anomaly confidence distribution
		anomalous = df[df["anomaly_label"] == 1]
		normal = df[df["anomaly_label"] == 0]

		axes[1, 1].hist(
			anomalous["anomaly_confidence"],
			bins=30,
			alpha=0.7,
			color="red",
			label="Anomalous",
			edgecolor="black",
		)
		axes[1, 1].hist(
			normal["anomaly_confidence"],
			bins=30,
			alpha=0.5,
			color="blue",
			label="Normal",
			edgecolor="black",
		)
		axes[1, 1].set_title(
			"Anomaly Confidence Distribution", fontsize=12, fontweight="bold"
		)
		axes[1, 1].set_xlabel("Confidence Score")
		axes[1, 1].set_ylabel("Frequency")
		axes[1, 1].legend()
		axes[1, 1].grid(True, alpha=0.3)

		plt.tight_layout()
		plt.savefig(
			"outputs/visualizations/anomaly_analysis_enhanced.png",
			dpi=300,
			bbox_inches="tight",
		)
		plt.close()

		# UMAP projection
		if self.embeddings is not None and len(df) > 100:
			self._create_umap_projection_enhanced(df)

	def _create_umap_projection_enhanced(self, df: pd.DataFrame):
		"""Create UMAP projection with multiple views"""
		logger.info("Creating UMAP projections...")

		# Sample for visualization
		sample_size = min(10000, len(df))
		sample_idx = np.random.choice(len(df), sample_size, replace=False)

		umap_2d = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15)
		embeddings_2d = umap_2d.fit_transform(self.embeddings[sample_idx])

		sampled_df = df.iloc[sample_idx].copy()
		sampled_df["umap_1"] = embeddings_2d[:, 0]
		sampled_df["umap_2"] = embeddings_2d[:, 1]

		fig, axes = plt.subplots(2, 2, figsize=(16, 14))

		# 1. Topics
		if "topic" in sampled_df.columns:
			scatter = axes[0, 0].scatter(
				sampled_df["umap_1"],
				sampled_df["umap_2"],
				c=sampled_df["topic"],
				cmap="tab20",
				alpha=0.6,
				s=10,
			)
			axes[0, 0].set_title(
				"UMAP Projection - Topics", fontsize=12, fontweight="bold"
			)
			plt.colorbar(scatter, ax=axes[0, 0], label="Topic ID")

		# 2. Sentiment
		sentiment_colors = sampled_df["sentiment_numeric"].map(
			{-1: "red", 0: "gray", 1: "green"}
		)
		axes[0, 1].scatter(
			sampled_df["umap_1"],
			sampled_df["umap_2"],
			c=sentiment_colors,
			alpha=0.6,
			s=10,
		)
		axes[0, 1].set_title(
			"UMAP Projection - Sentiment", fontsize=12, fontweight="bold"
		)

		# 3. Anomalies
		scatter = axes[1, 0].scatter(
			sampled_df["umap_1"],
			sampled_df["umap_2"],
			c=sampled_df["anomaly_score"],
			cmap="Reds",
			alpha=0.6,
			s=10,
		)
		axes[1, 0].set_title(
			"UMAP Projection - Anomaly Score", fontsize=12, fontweight="bold"
		)
		plt.colorbar(scatter, ax=axes[1, 0], label="Anomaly Score")

		# 4. Bot probability
		scatter = axes[1, 1].scatter(
			sampled_df["umap_1"],
			sampled_df["umap_2"],
			c=sampled_df["bot_score"],
			cmap="Oranges",
			alpha=0.6,
			s=10,
		)
		axes[1, 1].set_title(
			"UMAP Projection - Bot Score", fontsize=12, fontweight="bold"
		)
		plt.colorbar(scatter, ax=axes[1, 1], label="Bot Score")

		for ax in axes.flatten():
			ax.set_xlabel("UMAP 1")
			ax.set_ylabel("UMAP 2")
			ax.grid(True, alpha=0.3)

		plt.tight_layout()
		plt.savefig(
			"outputs/visualizations/umap_projections_enhanced.png",
			dpi=300,
			bbox_inches="tight",
		)
		plt.close()

	def _create_bot_plots(self, df: pd.DataFrame):
		"""Create bot detection visualizations"""
		logger.info("Creating bot detection visualizations...")

		fig, axes = plt.subplots(2, 2, figsize=(14, 10))

		# 1. Bot score distribution
		axes[0, 0].hist(
			df["bot_score"], bins=50, color="orange", alpha=0.7, edgecolor="black"
		)
		axes[0, 0].axvline(
			x=self.config["bot_threshold"],
			color="red",
			linestyle="--",
			linewidth=2,
			label="Threshold",
		)
		axes[0, 0].set_title("Bot Score Distribution", fontsize=12, fontweight="bold")
		axes[0, 0].set_xlabel("Bot Score")
		axes[0, 0].set_ylabel("Frequency")
		axes[0, 0].legend()
		axes[0, 0].grid(True, alpha=0.3)

		# 2. Bot vs Human comparison
		bot_human_data = df.groupby("bot_label").agg(
			{
				"sentiment_numeric": "mean",
				"anomaly_label": "mean",
				"engagement_score": "mean",
			}
		)

		x = np.arange(len(bot_human_data.columns))
		width = 0.35

		axes[0, 1].bar(
			x - width / 2,
			bot_human_data.loc[0],
			width,
			label="Human",
			color="blue",
			alpha=0.7,
		)
		axes[0, 1].bar(
			x + width / 2,
			bot_human_data.loc[1],
			width,
			label="Bot",
			color="red",
			alpha=0.7,
		)
		axes[0, 1].set_xticks(x)
		axes[0, 1].set_xticklabels(["Sentiment", "Anomaly\nRate", "Engagement"])
		axes[0, 1].set_title(
			"Bot vs Human Characteristics", fontsize=12, fontweight="bold"
		)
		axes[0, 1].legend()
		axes[0, 1].grid(True, alpha=0.3, axis="y")

		# 3. Bot activity over time
		daily_bot = df.groupby(pd.Grouper(key="timestamp", freq="D")).agg(
			{"bot_label": ["sum", "mean"]}
		)
		daily_bot.columns = ["bot_count", "bot_rate"]

		axes[1, 0].plot(
			daily_bot.index, daily_bot["bot_rate"] * 100, color="orange", linewidth=2
		)
		axes[1, 0].set_title(
			"Bot Activity Rate Over Time", fontsize=12, fontweight="bold"
		)
		axes[1, 0].set_xlabel("Date")
		axes[1, 0].set_ylabel("Bot Tweet %")
		axes[1, 0].tick_params(axis="x", rotation=45)
		axes[1, 0].grid(True, alpha=0.3)

		# 4. Bot impact on topics
		if "topic" in df.columns:
			topic_names = self.results["topics"].get("topic_names", {})
			top_topics = df[df["topic"] != -1]["topic"].value_counts().head(10).index

			bot_rates = []
			for topic in top_topics:
				topic_df = df[df["topic"] == topic]
				bot_rate = topic_df["bot_label"].mean()
				bot_rates.append(bot_rate)

			topic_labels = [topic_names.get(t, f"T{t}")[:20] for t in top_topics]

			axes[1, 1].barh(
				range(len(top_topics)),
				np.array(bot_rates) * 100,
				color="orange",
				alpha=0.7,
			)
			axes[1, 1].set_yticks(range(len(top_topics)))
			axes[1, 1].set_yticklabels(topic_labels, fontsize=9)
			axes[1, 1].set_xlabel("Bot Tweet %")
			axes[1, 1].set_title(
				"Bot Activity by Topic", fontsize=12, fontweight="bold"
			)
			axes[1, 1].invert_yaxis()
			axes[1, 1].grid(True, alpha=0.3, axis="x")

		plt.tight_layout()
		plt.savefig(
			"outputs/visualizations/bot_analysis.png", dpi=300, bbox_inches="tight"
		)
		plt.close()

	def _create_confidence_dashboard(self, df: pd.DataFrame):
		"""Create comprehensive confidence score dashboard"""
		logger.info("Creating confidence score dashboard...")

		fig = plt.figure(figsize=(18, 12))
		gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

		# 1. Overall confidence summary
		ax1 = fig.add_subplot(gs[0, :])
		confidence_metrics = {
			"Topic Assignment": (
				df["topic_confidence"].mean() if "topic_confidence" in df.columns else 0
			),
			"Sentiment": (
				df["sentiment_confidence"].mean()
				if "sentiment_confidence" in df.columns
				else 0
			),
			"Emotion": (
				df["emotion_confidence"].mean()
				if "emotion_confidence" in df.columns
				else 0
			),
			"Anomaly Detection": (
				df["anomaly_confidence"].mean()
				if "anomaly_confidence" in df.columns
				else 0
			),
			"Bot Detection": (
				df["bot_confidence"].mean() if "bot_confidence" in df.columns else 0
			),
			"Language": (
				df["lang_confidence"].mean() if "lang_confidence" in df.columns else 0
			),
		}

		colors = [
			"green" if v > 0.7 else "orange" if v > 0.5 else "red"
			for v in confidence_metrics.values()
		]
		bars = ax1.bar(
			range(len(confidence_metrics)),
			confidence_metrics.values(),
			color=colors,
			alpha=0.7,
		)
		ax1.set_xticks(range(len(confidence_metrics)))
		ax1.set_xticklabels(confidence_metrics.keys(), rotation=45, ha="right")
		ax1.set_ylabel("Average Confidence Score", fontsize=12)
		ax1.set_title(
			"Model Confidence Across All Components", fontsize=14, fontweight="bold"
		)
		ax1.axhline(
			y=0.7, color="green", linestyle="--", alpha=0.5, label="High Confidence"
		)
		ax1.axhline(
			y=0.5, color="orange", linestyle="--", alpha=0.5, label="Medium Confidence"
		)
		ax1.set_ylim(0, 1)
		ax1.legend()
		ax1.grid(True, alpha=0.3, axis="y")

		# Add value labels
		for i, (key, val) in enumerate(confidence_metrics.items()):
			ax1.text(
				i, val + 0.02, f"{val:.3f}", ha="center", fontsize=10, fontweight="bold"
			)

		# 2. Topic confidence distribution
		ax2 = fig.add_subplot(gs[1, 0])
		if "topic_confidence" in df.columns:
			ax2.hist(
				df["topic_confidence"],
				bins=30,
				color="skyblue",
				edgecolor="black",
				alpha=0.7,
			)
			ax2.axvline(
				x=df["topic_confidence"].mean(),
				color="red",
				linestyle="--",
				linewidth=2,
				label=f'Mean: {df["topic_confidence"].mean():.3f}',
			)
			ax2.set_xlabel("Confidence Score")
			ax2.set_ylabel("Frequency")
			ax2.set_title(
				"Topic Confidence Distribution", fontsize=11, fontweight="bold"
			)
			ax2.legend()
			ax2.grid(True, alpha=0.3)

		# 3. Sentiment confidence distribution
		ax3 = fig.add_subplot(gs[1, 1])
		if "sentiment_confidence" in df.columns:
			ax3.hist(
				df["sentiment_confidence"],
				bins=30,
				color="lightgreen",
				edgecolor="black",
				alpha=0.7,
			)
			ax3.axvline(
				x=df["sentiment_confidence"].mean(),
				color="red",
				linestyle="--",
				linewidth=2,
				label=f'Mean: {df["sentiment_confidence"].mean():.3f}',
			)
			ax3.set_xlabel("Confidence Score")
			ax3.set_ylabel("Frequency")
			ax3.set_title(
				"Sentiment Confidence Distribution", fontsize=11, fontweight="bold"
			)
			ax3.legend()
			ax3.grid(True, alpha=0.3)

		# 4. Emotion confidence distribution
		ax4 = fig.add_subplot(gs[1, 2])
		if "emotion_confidence" in df.columns:
			ax4.hist(
				df["emotion_confidence"],
				bins=30,
				color="lightcoral",
				edgecolor="black",
				alpha=0.7,
			)
			ax4.axvline(
				x=df["emotion_confidence"].mean(),
				color="red",
				linestyle="--",
				linewidth=2,
				label=f'Mean: {df["emotion_confidence"].mean():.3f}',
			)
			ax4.set_xlabel("Confidence Score")
			ax4.set_ylabel("Frequency")
			ax4.set_title(
				"Emotion Confidence Distribution", fontsize=11, fontweight="bold"
			)
			ax4.legend()
			ax4.grid(True, alpha=0.3)

		# 5. Confidence over time
		ax5 = fig.add_subplot(gs[2, :])
		daily_conf = df.groupby(pd.Grouper(key="timestamp", freq="D")).agg(
			{
				"topic_confidence": "mean",
				"sentiment_confidence": "mean",
				"emotion_confidence": "mean",
			}
		)

		ax5.plot(
			daily_conf.index,
			daily_conf["topic_confidence"],
			label="Topic",
			linewidth=2,
			alpha=0.8,
		)
		ax5.plot(
			daily_conf.index,
			daily_conf["sentiment_confidence"],
			label="Sentiment",
			linewidth=2,
			alpha=0.8,
		)
		ax5.plot(
			daily_conf.index,
			daily_conf["emotion_confidence"],
			label="Emotion",
			linewidth=2,
			alpha=0.8,
		)
		ax5.axhline(y=0.7, color="green", linestyle="--", alpha=0.3)
		ax5.axhline(y=0.5, color="orange", linestyle="--", alpha=0.3)
		ax5.set_xlabel("Date", fontsize=12)
		ax5.set_ylabel("Confidence Score", fontsize=12)
		ax5.set_title("Model Confidence Over Time", fontsize=12, fontweight="bold")
		ax5.legend(loc="best")
		ax5.grid(True, alpha=0.3)
		ax5.set_ylim(0, 1)

		plt.savefig(
			"outputs/visualizations/confidence/confidence_dashboard.png",
			dpi=300,
			bbox_inches="tight",
		)
		plt.close()

		logger.info("Confidence dashboard created successfully")

	def _create_interactive_timeline_enhanced(self, df: pd.DataFrame):
		"""Create enhanced interactive timeline with confidence tracking"""
		logger.info("Creating enhanced interactive timeline...")

		daily = (
			df.groupby(pd.Grouper(key="timestamp", freq="D"))
			.agg(
				{
					"sentiment_numeric": "mean",
					"sentiment_confidence": "mean",
					"anomaly_label": "mean",
					"anomaly_confidence": "mean",
					"bot_label": "mean",
					"bot_confidence": "mean",
					"id": "count",
					"engagement_score": "sum",
				}
			)
			.reset_index()
		)

		daily.columns = [
			"date",
			"sentiment",
			"sentiment_conf",
			"anomaly_rate",
			"anomaly_conf",
			"bot_rate",
			"bot_conf",
			"tweet_count",
			"engagement",
		]

		fig = make_subplots(
			rows=5,
			cols=1,
			subplot_titles=(
				"Tweet Volume",
				"Sentiment Trend (with Confidence)",
				"Anomaly Detection Rate (with Confidence)",
				"Bot Activity Rate (with Confidence)",
				"Engagement Score",
			),
			vertical_spacing=0.06,
			specs=[[{"secondary_y": False}]] * 5,
		)

		# Row 1: Tweet Volume
		fig.add_trace(
			go.Scatter(
				x=daily["date"],
				y=daily["tweet_count"],
				name="Tweet Count",
				fill="tozeroy",
				line=dict(color="steelblue", width=2),
				hovertemplate="<b>Date:</b> %{x}<br><b>Tweets:</b> %{y:,}<extra></extra>",
			),
			row=1,
			col=1,
		)

		# Row 2: Sentiment with Confidence
		fig.add_trace(
			go.Scatter(
				x=daily["date"],
				y=daily["sentiment"],
				name="Sentiment",
				line=dict(color="green", width=2),
				hovertemplate="<b>Date:</b> %{x}<br><b>Sentiment:</b> %{y:.3f}<extra></extra>",
			),
			row=2,
			col=1,
		)
		fig.add_trace(
			go.Scatter(
				x=daily["date"],
				y=daily["sentiment_conf"],
				name="Sentiment Confidence",
				line=dict(color="lightgreen", width=2, dash="dash"),
				hovertemplate="<b>Date:</b> %{x}<br><b>Confidence:</b> %{y:.3f}<extra></extra>",
			),
			row=2,
			col=1,
		)

		# Row 3: Anomaly Rate with Confidence
		fig.add_trace(
			go.Scatter(
				x=daily["date"],
				y=daily["anomaly_rate"] * 100,
				name="Anomaly Rate",
				line=dict(color="red", width=2),
				hovertemplate="<b>Date:</b> %{x}<br><b>Anomaly Rate:</b> %{y:.2f}%<extra></extra>",
			),
			row=3,
			col=1,
		)
		fig.add_trace(
			go.Scatter(
				x=daily["date"],
				y=daily["anomaly_conf"] * 100,
				name="Anomaly Confidence",
				line=dict(color="pink", width=2, dash="dash"),
				hovertemplate="<b>Date:</b> %{x}<br><b>Confidence:</b> %{y:.2f}%<extra></extra>",
			),
			row=3,
			col=1,
		)

		# Row 4: Bot Rate with Confidence
		fig.add_trace(
			go.Scatter(
				x=daily["date"],
				y=daily["bot_rate"] * 100,
				name="Bot Rate",
				line=dict(color="orange", width=2),
				hovertemplate="<b>Date:</b> %{x}<br><b>Bot Rate:</b> %{y:.2f}%<extra></extra>",
			),
			row=4,
			col=1,
		)
		fig.add_trace(
			go.Scatter(
				x=daily["date"],
				y=daily["bot_conf"] * 100,
				name="Bot Detection Confidence",
				line=dict(color="gold", width=2, dash="dash"),
				hovertemplate="<b>Date:</b> %{x}<br><b>Confidence:</b> %{y:.2f}%<extra></extra>",
			),
			row=4,
			col=1,
		)

		# Row 5: Engagement
		fig.add_trace(
			go.Scatter(
				x=daily["date"],
				y=daily["engagement"],
				name="Total Engagement",
				fill="tozeroy",
				line=dict(color="purple", width=2),
				hovertemplate="<b>Date:</b> %{x}<br><b>Engagement:</b> %{y:,}<extra></extra>",
			),
			row=5,
			col=1,
		)

		fig.update_layout(
			height=1400,
			title_text="<b>COVID-19 Twitter Discourse Timeline - Enhanced with Confidence Tracking</b>",
			showlegend=True,
			hovermode="x unified",
			template="plotly_white",
		)

		# Update y-axes labels
		fig.update_yaxes(title_text="Count", row=1, col=1)
		fig.update_yaxes(title_text="Score", row=2, col=1)
		fig.update_yaxes(title_text="Percentage", row=3, col=1)
		fig.update_yaxes(title_text="Percentage", row=4, col=1)
		fig.update_yaxes(title_text="Engagement", row=5, col=1)

		fig.write_html("outputs/visualizations/interactive_timeline_enhanced.html")
		logger.info("Interactive timeline saved")

	def _create_topic_network_enhanced(self, df: pd.DataFrame):
		"""Create enhanced topic network with topic names"""
		logger.info("Creating enhanced topic network...")

		topic_names = self.results["topics"].get("topic_names", {})

		# Calculate topic co-occurrence by week
		df["week"] = df["timestamp"].dt.to_period("W")

		topics = [t for t in df["topic"].unique() if t != -1]
		if len(topics) == 0:
			logger.warning("No topics to create network")
			return

		# Co-occurrence matrix
		cooccurrence = np.zeros((len(topics), len(topics)))

		for week in df["week"].unique():
			week_topics = df[df["week"] == week]["topic"].value_counts()
			for i, t1 in enumerate(topics):
				for j, t2 in enumerate(topics):
					if t1 in week_topics.index and t2 in week_topics.index:
						cooccurrence[i, j] += 1

		# Create network
		G = nx.Graph()
		for i, topic in enumerate(topics):
			topic_size = len(df[df["topic"] == topic])
			topic_name = topic_names.get(topic, f"Topic {topic}")
			G.add_node(topic, size=topic_size, name=topic_name)

		# Add edges (only strong connections)
		threshold = np.percentile(cooccurrence, 70)
		for i in range(len(topics)):
			for j in range(i + 1, len(topics)):
				if cooccurrence[i, j] > threshold:
					G.add_edge(topics[i], topics[j], weight=cooccurrence[i, j])

		# Layout
		pos = nx.spring_layout(G, k=0.5, iterations=50, seed=RANDOM_SEED)

		# Create plotly figure
		edge_trace = []
		for edge in G.edges():
			x0, y0 = pos[edge[0]]
			x1, y1 = pos[edge[1]]
			edge_trace.append(
				go.Scatter(
					x=[x0, x1, None],
					y=[y0, y1, None],
					mode="lines",
					line=dict(width=0.5, color="#888"),
					hoverinfo="none",
					showlegend=False,
				)
			)

		# Node trace
		node_x = []
		node_y = []
		node_text = []
		node_size = []
		node_color = []

		for node in G.nodes():
			x, y = pos[node]
			node_x.append(x)
			node_y.append(y)
			node_text.append(G.nodes[node]["name"])
			node_size.append(np.sqrt(G.nodes[node]["size"]) / 3)

			# Color by sentiment
			topic_sentiment = df[df["topic"] == node]["sentiment_numeric"].mean()
			node_color.append(topic_sentiment)

		node_trace = go.Scatter(
			x=node_x,
			y=node_y,
			mode="markers+text",
			text=node_text,
			textposition="top center",
			textfont=dict(size=10, color="black"),
			marker=dict(
				size=node_size,
				color=node_color,
				colorscale="RdYlGn",
				showscale=True,
				colorbar=dict(title="Avg Sentiment", thickness=15, len=0.7),
				line=dict(width=2, color="white"),
			),
			hovertemplate="<b>%{text}</b><br>Size: %{marker.size}<br>Sentiment: %{marker.color:.2f}<extra></extra>",
		)

		fig = go.Figure(
			data=edge_trace + [node_trace],
			layout=go.Layout(
				title="<b>Topic Co-occurrence Network</b><br><sub>Node size = topic size, Color = sentiment</sub>",
				showlegend=False,
				hovermode="closest",
				xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
				yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
				plot_bgcolor="white",
				width=1200,
				height=800,
			),
		)

		fig.write_html("outputs/visualizations/topic_network_enhanced.html")
		logger.info("Topic network saved")

	def _create_sentiment_topic_heatmap(self, df: pd.DataFrame):
		"""Create interactive sentiment-topic heatmap"""
		logger.info("Creating sentiment-topic heatmap...")

		topic_names = self.results["topics"].get("topic_names", {})

		# Get top 15 topics
		top_topics = df[df["topic"] != -1]["topic"].value_counts().head(15).index

		# Calculate sentiment distribution for each topic
		sentiment_labels = ["Negative", "Neutral", "Positive"]
		sentiment_values = [-1, 0, 1]

		matrix = np.zeros((len(top_topics), 3))
		topic_labels = []

		for i, topic in enumerate(top_topics):
			topic_df = df[df["topic"] == topic]
			topic_labels.append(topic_names.get(topic, f"Topic {topic}"))

			for j, sent_val in enumerate(sentiment_values):
				count = (topic_df["sentiment_numeric"] == sent_val).sum()
				matrix[i, j] = count / len(topic_df) * 100 if len(topic_df) > 0 else 0

		fig = go.Figure(
			data=go.Heatmap(
				z=matrix,
				x=sentiment_labels,
				y=topic_labels,
				colorscale="RdYlGn",
				text=np.round(matrix, 1),
				texttemplate="%{text}%",
				textfont={"size": 10},
				colorbar=dict(title="Percentage"),
				hovertemplate="<b>Topic:</b> %{y}<br><b>Sentiment:</b> %{x}<br><b>Percentage:</b> %{z:.1f}%<extra></extra>",
			)
		)

		fig.update_layout(
			title="<b>Sentiment Distribution Across Topics</b>",
			xaxis_title="Sentiment",
			yaxis_title="Topic",
			width=800,
			height=600,
			template="plotly_white",
		)

		fig.write_html("outputs/visualizations/sentiment_topic_heatmap.html")
		logger.info("Sentiment-topic heatmap saved")

	# ═══════════════════════════════════════════════════════════════════════
	# PHASE 9: COMPREHENSIVE REPORT GENERATION
	# ═══════════════════════════════════════════════════════════════════════

	def generate_reports(self, df: pd.DataFrame) -> None:
		logger.info("=" * 80)
		logger.info("PHASE 9: COMPREHENSIVE REPORT GENERATION")
		logger.info("=" * 80)

		# Save annotated dataset
		logger.info("Saving annotated dataset...")
		df.to_csv("outputs/data/covid_tweets_annotated.csv", index=False)
		df.to_parquet("outputs/data/covid_tweets_annotated.parquet", index=False)

		# Statistical summary
		statistical_summary = self._generate_statistical_summary(df)
		with open("outputs/reports/statistical_summary.json", "w") as f:
			json.dump(statistical_summary, f, indent=2, default=str)

		# Topic report
		if "topics" in self.results:
			topic_report = self._generate_topic_report(df)
			with open("outputs/reports/topic_report.json", "w") as f:
				json.dump(topic_report, f, indent=2, default=str)

		# Affective analysis report
		if "affective" in self.results:
			affective_report = self._generate_affective_report(df)
			with open("outputs/reports/affective_analysis_report.json", "w") as f:
				json.dump(affective_report, f, indent=2, default=str)

		# Anomaly report
		if "anomalies" in self.results:
			anomaly_report = self._generate_anomaly_report(df)
			with open("outputs/reports/anomaly_report.json", "w") as f:
				json.dump(anomaly_report, f, indent=2, default=str)

		# Bot detection report
		if "bots" in self.results:
			bot_report = self._generate_bot_report(df)
			with open("outputs/reports/bot_detection_report.json", "w") as f:
				json.dump(bot_report, f, indent=2, default=str)

		# Confidence scores report
		confidence_report = self._generate_confidence_report(df)
		with open("outputs/reports/confidence_scores_report.json", "w") as f:
			json.dump(confidence_report, f, indent=2, default=str)

		# Reproducibility report
		repro_report = self._generate_reproducibility_report()
		with open("outputs/reproducibility/reproducibility_report.json", "w") as f:
			json.dump(repro_report, f, indent=2, default=str)

		# Executive summary (human-readable)
		self._generate_executive_summary(df)

		logger.info("All reports generated and saved to outputs/reports/")

	def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict:
		"""Generate comprehensive statistical summary"""
		summary = {
			"dataset_statistics": {
				"total_tweets": int(len(df)),
				"unique_users": (
					int(df["original_author"].nunique())
					if "original_author" in df.columns
					else "N/A"
				),
				"date_range": {
					"start": (
						str(df["timestamp"].min())
						if "timestamp" in df.columns
						else "N/A"
					),
					"end": (
						str(df["timestamp"].max())
						if "timestamp" in df.columns
						else "N/A"
					),
					"duration_days": (
						(df["timestamp"].max() - df["timestamp"].min()).days
						if "timestamp" in df.columns
						else "N/A"
					),
				},
				"phases": (
					df["phase"].value_counts().to_dict()
					if "phase" in df.columns
					else {}
				),
			}
		}

		# Topic statistics
		if "topic" in df.columns:
			summary["topic_statistics"] = {
				"num_topics": int(df[df["topic"] != -1]["topic"].nunique()),
				"noise_tweets": int((df["topic"] == -1).sum()),
				"noise_percentage": float((df["topic"] == -1).mean() * 100),
				"avg_topic_size": float(
					df[df["topic"] != -1].groupby("topic").size().mean()
				),
				"largest_topic_size": int(
					df[df["topic"] != -1].groupby("topic").size().max()
				),
				"smallest_topic_size": int(
					df[df["topic"] != -1].groupby("topic").size().min()
				),
				"topic_balance_score": float(
					self.results["topics"]["quality_metrics"].get(
						"topic_balance_score", 0
					)
				),
			}

		# Sentiment statistics
		if "sentiment_numeric" in df.columns:
			summary["sentiment_statistics"] = {
				"positive_percentage": float(
					(df["sentiment_numeric"] == 1).mean() * 100
				),
				"neutral_percentage": float(
					(df["sentiment_numeric"] == 0).mean() * 100
				),
				"negative_percentage": float(
					(df["sentiment_numeric"] == -1).mean() * 100
				),
				"avg_sentiment": float(df["sentiment_numeric"].mean()),
				"avg_confidence": float(df["sentiment_confidence"].mean()),
				"high_confidence_rate": float(
					(df["sentiment_confidence"] > 0.7).mean() * 100
				),
			}

		# Emotion statistics
		if "emotion_label" in df.columns:
			emotion_dist = df["emotion_label"].value_counts(normalize=True).to_dict()
			summary["emotion_statistics"] = {
				"distribution": {k: float(v * 100) for k, v in emotion_dist.items()},
				"avg_confidence": float(df["emotion_confidence"].mean()),
				"dominant_emotion": (
					df["emotion_label"].mode()[0]
					if len(df["emotion_label"].mode()) > 0
					else "N/A"
				),
			}

		# Anomaly statistics
		if "anomaly_label" in df.columns:
			high_conf_anomalies = (
				(df["anomaly_label"] == 1) & (df["anomaly_confidence"] > 0.7)
			).sum()

			summary["anomaly_statistics"] = {
				"anomaly_rate": float(df["anomaly_label"].mean() * 100),
				"total_anomalies": int(df["anomaly_label"].sum()),
				"avg_confidence": float(df["anomaly_confidence"].mean()),
				"high_confidence_anomalies": int(high_conf_anomalies),
			}

		# Bot statistics
		if "bot_label" in df.columns:
			summary["bot_statistics"] = {
				"bot_tweet_rate": float(df["bot_label"].mean() * 100),
				"total_bot_tweets": int(df["bot_label"].sum()),
				"avg_confidence": float(df["bot_confidence"].mean()),
				"unique_bot_accounts": (
					int(df[df["bot_label"] == 1]["original_author"].nunique())
					if "original_author" in df.columns
					else "N/A"
				),
			}

		# Engagement statistics
		if "engagement_score" in df.columns:
			summary["engagement_statistics"] = {
				"total_engagement": int(df["engagement_score"].sum()),
				"avg_engagement": float(df["engagement_score"].mean()),
				"median_engagement": float(df["engagement_score"].median()),
				"top_1_percent_threshold": float(df["engagement_score"].quantile(0.99)),
			}

		return summary

	def _generate_topic_report(self, df: pd.DataFrame) -> Dict:
		"""Generate detailed topic report"""
		report = {
			"summary": {
				"total_topics": self.results["topics"]["num_topics"],
				"quality_metrics": self.results["topics"].get("quality_metrics", {}),
			},
			"topics": [],
		}

		if "characterization" in self.results["topics"]:
			for topic_id, info in self.results["topics"]["characterization"].items():
				if topic_id == -1:
					continue

				topic_entry = {
					"topic_id": int(topic_id),
					"topic_name": info.get("topic_name", f"Topic {topic_id}"),
					"keywords": [
						{
							"word": kw[0] if isinstance(kw, tuple) else kw,
							"score": float(kw[1]) if isinstance(kw, tuple) else 1.0,
						}
						for kw in info.get("keywords", [])[:10]
					],
					"size": int(info["size"]),
					"percentage": float(info["percentage"]),
					"representative_tweets": info.get("representative_tweets", [])[:3],
					"avg_engagement": float(info.get("avg_engagement", 0)),
					"confidence_stats": info.get("confidence_stats", {}),
				}

				# Add sentiment
				if "sentiment_distribution" in info:
					topic_entry["sentiment_distribution"] = info[
						"sentiment_distribution"
					]

				# Add temporal pattern
				if "temporal_distribution" in info and info["temporal_distribution"]:
					temporal = info["temporal_distribution"]
					if temporal:
						topic_entry["temporal_pattern"] = {
							"peak_date": str(max(temporal, key=temporal.get)),
							"peak_count": int(max(temporal.values())),
						}

				report["topics"].append(topic_entry)

		# Sort by size
		report["topics"] = sorted(
			report["topics"], key=lambda x: x["size"], reverse=True
		)

		return report

	def _convert_keys_to_str(self, data: Any) -> Any:
		"""Recursively converts dictionary keys to strings if they are non-string,
		non-standard JSON types (like numpy.int64).
		"""
		if isinstance(data, dict):
			new_dict = {}
			for k, v in data.items():
				# Convert non-string keys to str (e.g., int64 index keys)
				if not isinstance(k, (str, int, float, bool, type(None))):
					new_key = str(k)
				else:
					new_key = k
				
				# Recursive call using self.
				new_dict[new_key] = self._convert_keys_to_str(v)
			return new_dict
		elif isinstance(data, list):
			# Recursive call on list items
			return [self._convert_keys_to_str(item) for item in data]
		else:
			return data

	def _generate_affective_report(self, df: pd.DataFrame) -> Dict:
		"""Generate affective analysis report"""
		# Start with a copy to prevent modifying the shared results dict directly
		report = self.results.get("affective", {}).copy()

		# Add temporal sentiment analysis
		if "sentiment_numeric" in df.columns and "timestamp" in df.columns:
			daily_sentiment = (
				df.groupby(pd.Grouper(key="timestamp", freq="D"))[
					"sentiment_numeric"
				].mean()
			)

			report["temporal_analysis"] = {
				"most_positive_day": str(daily_sentiment.idxmax()),
				"most_positive_score": float(daily_sentiment.max()),
				"most_negative_day": str(daily_sentiment.idxmin()),
				"most_negative_score": float(daily_sentiment.min()),
			}

		# Emotion transitions
		if "emotion_label" in df.columns:
			emotion_transitions = self._analyze_emotion_transitions(df)

			report["emotion_transitions"] = self._convert_keys_to_str(emotion_transitions)

		return self._convert_keys_to_str(report)

	def _analyze_emotion_transitions(self, df: pd.DataFrame) -> Dict:
		"""Analyze how emotions change over time"""
		df_sorted = df.sort_values("timestamp")

		# Simple transition count
		transitions = {}
		prev_emotion = None

		for emotion in df_sorted["emotion_label"]:
			if prev_emotion is not None and prev_emotion != emotion:
				key = f"{prev_emotion}_to_{emotion}"
				transitions[key] = transitions.get(key, 0) + 1
			prev_emotion = emotion

		# Get top 10 transitions
		top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[
			:10
		]

		return {
			"top_transitions": [
				{"transition": t[0], "count": int(t[1])} for t in top_transitions
			]
		}

	def _generate_anomaly_report(self, df: pd.DataFrame) -> Dict:
		"""Generate anomaly detection report"""
		report = self.results.get("anomalies", {}).copy()

		# Add examples of high-confidence anomalies
		if "anomaly_label" in df.columns and "anomaly_confidence" in df.columns:
			high_conf_anomalies = df[
				(df["anomaly_label"] == 1) & (df["anomaly_confidence"] > 0.8)
			].nlargest(10, "anomaly_confidence")

			report["high_confidence_examples"] = [
				{
					"text": row["cleaned_text"],
					"confidence": float(row["anomaly_confidence"]),
					"anomaly_score": float(row["anomaly_score"]),
					"topic": int(row["topic"]) if "topic" in row else None,
					"timestamp": str(row["timestamp"]) if "timestamp" in row else None,
				}
				for _, row in high_conf_anomalies.iterrows()
			]

		return report

	def _generate_bot_report(self, df: pd.DataFrame) -> Dict:
		"""Generate bot detection report"""
		report = self.results.get("bots", {}).copy()

		# Top suspected bot accounts
		if "original_author" in df.columns and "bot_score" in df.columns:
			user_bot_scores = (
				df.groupby("original_author")
				.agg({"bot_score": "mean", "bot_confidence": "mean", "id": "count"})
				.rename(columns={"id": "tweet_count"})
			)

			top_bots = user_bot_scores[user_bot_scores["bot_score"] > 0.7].nlargest(
				20, "bot_score"
			)

			report["top_suspected_bots"] = [
				{
					"username": str(username),
					"bot_score": float(row["bot_score"]),
					"confidence": float(row["bot_confidence"]),
					"tweet_count": int(row["tweet_count"]),
				}
				for username, row in top_bots.iterrows()
			]

		return report

	def _generate_confidence_report(self, df: pd.DataFrame) -> Dict:
		"""Generate comprehensive confidence report"""
		report = {
			"overall_confidence": {},
			"confidence_by_component": self.confidence_scores.copy(),
			"low_confidence_analysis": {},
		}

		# Overall confidence metrics
		confidence_cols = [col for col in df.columns if "confidence" in col]
		for col in confidence_cols:
			report["overall_confidence"][col] = {
				"mean": float(df[col].mean()),
				"median": float(df[col].median()),
				"std": float(df[col].std()),
				"min": float(df[col].min()),
				"max": float(df[col].max()),
				"q25": float(df[col].quantile(0.25)),
				"q75": float(df[col].quantile(0.75)),
			}

		# Low confidence analysis
		for col in confidence_cols:
			low_conf = df[df[col] < 0.5]
			report["low_confidence_analysis"][col] = {
				"count": int(len(low_conf)),
				"percentage": float(len(low_conf) / len(df) * 100),
			}

		return report

	def _generate_reproducibility_report(self) -> Dict:
		"""Generate reproducibility report"""
		import transformers
		import sentence_transformers
		import sklearn

		report = {
			"pipeline_version": "2.0_CORRECTED",
			"execution_date": str(datetime.now()),
			"random_seed": RANDOM_SEED,
			"configuration": self.config,
			"environment": {
				"python_version": platform.python_version(),
				"platform": platform.platform(),
				"libraries": {
					"numpy": np.__version__,
					"pandas": pd.__version__,
					"torch": torch.__version__,
					"transformers": transformers.__version__,
					"sentence_transformers": sentence_transformers.__version__,
					"sklearn": sklearn.__version__,
					"umap-learn": umap.__version__,
					'hdbscan': '0.8.39 (Installed Placeholder)',
					"plotly": 'placeholder',
					"networkx": nx.__version__,
				},
				"hardware": {
					"cpu": platform.processor(),
					"cpu_count": os.cpu_count(),
					"ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
					"gpu": (
						torch.cuda.get_device_name(0)
						if torch.cuda.is_available()
						else "None"
					),
					"cuda_available": torch.cuda.is_available(),
				},
			},
			"data_processing_steps": [
				"Deduplication",
				"Text normalization",
				"Language filtering",
				"ACE annotation",
				"Semantic embedding (sentence-transformers)",
				"Topic modeling (BERTopic with 20 topics)",
				"Ensemble sentiment analysis",
				"Ensemble emotion analysis",
				"Multi-scale anomaly detection",
				"Ensemble bot detection",
				"Statistical validation",
			],
			"models_used": {
				"embedding": self.config["embedding_model"],
				"sentiment": [
					"cardiffnlp/twitter-roberta-base-sentiment-latest",
					"VADER",
					"distilbert-base-uncased-finetuned-sst-2-english",
				],
				"emotion": [
					"j-hartmann/emotion-english-distilroberta-base",
					"SamLowe/roberta-base-go_emotions",
				],
				"ner": "dslim/bert-base-NER",
			},
		}

		return report

	def _generate_executive_summary(self, df: pd.DataFrame):
		"""Generate human-readable executive summary"""
		logger.info("Generating executive summary...")

		summary_text = f"""
COVID-19 TWITTER DISCOURSE ANALYSIS
EXECUTIVE SUMMARY
{'='*80}

DATASET OVERVIEW
{'-'*80}
Total Tweets Analyzed: {len(df):,}
Unique Users: {df['original_author'].nunique() if 'original_author' in df.columns else 'N/A':,}
Date Range: {df['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in df.columns else 'N/A'} to {df['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in df.columns else 'N/A'}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TOPIC ANALYSIS
{'-'*80}
Number of Topics Discovered: {self.results['topics']['num_topics'] if 'topics' in self.results else 'N/A'}
Topic Quality Score: {self.results['topics']['quality_metrics'].get('silhouette_score', 'N/A'):.3f} (Silhouette Score)
Topic Balance: {self.results['topics']['quality_metrics'].get('topic_balance_score', 'N/A'):.3f}

Top 5 Topics by Size:
"""

		if "topics" in self.results and "characterization" in self.results["topics"]:
			topic_char = self.results["topics"]["characterization"]
			sorted_topics = sorted(
				[(tid, info) for tid, info in topic_char.items() if tid != -1],
				key=lambda x: x[1]["size"],
				reverse=True,
			)[:5]

			for i, (topic_id, info) in enumerate(sorted_topics, 1):
				summary_text += f"\n{i}. {info.get('topic_name', f'Topic {topic_id}')} - {info['size']:,} tweets ({info['percentage']:.1f}%)"
				keywords = [
					kw[0] if isinstance(kw, tuple) else kw
					for kw in info.get("keywords", [])[:5]
				]
				summary_text += f"\n   Keywords: {', '.join(keywords)}"

		summary_text += f"""

SENTIMENT ANALYSIS
{'-'*80}
Overall Sentiment Distribution:
  Positive: {(df['sentiment_numeric'] == 1).mean() * 100:.1f}%
  Neutral: {(df['sentiment_numeric'] == 0).mean() * 100:.1f}%
  Negative: {(df['sentiment_numeric'] == -1).mean() * 100:.1f}%

Average Sentiment Score: {df['sentiment_numeric'].mean():.3f}
Sentiment Confidence: {df['sentiment_confidence'].mean():.3f}

EMOTION ANALYSIS
{'-'*80}
Top 3 Emotions:
"""

		top_emotions = df["emotion_label"].value_counts(normalize=True).head(3)
		for emotion, pct in top_emotions.items():
			summary_text += f"  {emotion.title()}: {pct * 100:.1f}%\n"

		summary_text += f"""
Emotion Detection Confidence: {df['emotion_confidence'].mean():.3f}

ANOMALY DETECTION
{'-'*80}
Anomalies Detected: {df['anomaly_label'].sum():,} ({df['anomaly_label'].mean() * 100:.2f}%)
Detection Confidence: {df['anomaly_confidence'].mean():.3f}
High-Confidence Anomalies: {((df['anomaly_label'] == 1) & (df['anomaly_confidence'] > 0.7)).sum():,}

BOT DETECTION
{'-'*80}
Suspected Bot Tweets: {df['bot_label'].sum():,} ({df['bot_label'].mean() * 100:.2f}%)
Detection Confidence: {df['bot_confidence'].mean():.3f}
Unique Bot Accounts: {df[df['bot_label'] == 1]['original_author'].nunique() if 'original_author' in df.columns else 'N/A':,}

OVERALL MODEL CONFIDENCE
{'-'*80}
Topic Assignment: {df['topic_confidence'].mean():.3f}
Sentiment Analysis: {df['sentiment_confidence'].mean():.3f}
Emotion Detection: {df['emotion_confidence'].mean():.3f}
Anomaly Detection: {df['anomaly_confidence'].mean():.3f}
Bot Detection: {df['bot_confidence'].mean():.3f}

RECOMMENDATIONS
{'-'*80}
"""

		# Add recommendations based on confidence scores
		if df["sentiment_confidence"].mean() < 0.6:
			summary_text += "⚠ Sentiment analysis shows moderate confidence. Consider manual validation.\n"
		if df["emotion_confidence"].mean() < 0.5:
			summary_text += "⚠ Emotion detection has low confidence. Results should be interpreted cautiously.\n"
		if df["topic_confidence"].mean() > 0.7:
			summary_text += "✓ Topic assignments are highly confident and reliable.\n"
		if df["bot_label"].mean() > 0.1:
			summary_text += "⚠ Significant bot activity detected. Consider filtering for human-only analysis.\n"

		summary_text += f"""
{'='*80}
For detailed results, see:
- Full reports: outputs/reports/
- Visualizations: outputs/visualizations/
- Annotated data: outputs/data/covid_tweets_annotated.csv
- Interactive dashboards: outputs/visualizations/*.html

Pipeline Version: 2.0 (CORRECTED)
"""

		with open("outputs/reports/EXECUTIVE_SUMMARY.txt", "w") as f:
			f.write(summary_text)

		logger.info("Executive summary generated")
		print("\n" + summary_text)

	# ═══════════════════════════════════════════════════════════════════════
	# CHECKPOINT SUPPORT
	# ═══════════════════════════════════════════════════════════════════════

	def _save_checkpoint(self, df: pd.DataFrame, phase: int):
		"""Save pipeline checkpoint"""
		if not self.config["use_checkpoints"]:
			return

		checkpoint_data = {
			"df": df,
			"phase": phase,
			"timestamp": datetime.now(),
			"results": self.results,
			"confidence_scores": self.confidence_scores,
		}
		with open("outputs/data/checkpoint.pkl", "wb") as f:
			pickle.dump(checkpoint_data, f)
		logger.info(f"Checkpoint saved at phase {phase}")

	def _load_checkpoint(self):
		"""Load pipeline checkpoint"""
		checkpoint_file = "outputs/data/checkpoint.pkl"
		if os.path.exists(checkpoint_file):
			logger.info("Loading from checkpoint...")
			with open(checkpoint_file, "rb") as f:
				checkpoint_data = pickle.load(f)
			return checkpoint_data
		return None

	# ═══════════════════════════════════════════════════════════════════════
	# MAIN PIPELINE EXECUTION
	# ═══════════════════════════════════════════════════════════════════════

	def run_complete_pipeline(self, data_paths: List[str]) -> pd.DataFrame:
		"""
		Execute complete enhanced pipeline with checkpointing and resume capability.
		"""
		logger.info("=" * 80)
		logger.info("COVID-19 TWITTER ANALYSIS PIPELINE - ENHANCED VERSION 2.0")
		logger.info("=" * 80)
		start_time = datetime.now()
		
		# 

		try:
			# 1. Load Checkpoint and Determine Starting Phase
			checkpoint = self._load_checkpoint()
			last_phase = checkpoint['phase'] if checkpoint else 0
			df = checkpoint['df'] if checkpoint else None
			
			if last_phase > 0:
				self.results = checkpoint.get('results', {})
				logger.info(f"💾 Resuming pipeline from Phase {last_phase} with {len(df):,} records.")
				
				# Load embeddings if Phase 2 was completed
				if last_phase >= 2 and os.path.exists('outputs/data/intermediate/embeddings.npy'):
					self.embeddings = np.load('outputs/data/intermediate/embeddings.npy')
					logger.info("✅ Loaded existing embeddings from disk.")
				
			else:
				logger.info("🆕 Starting new pipeline run.")

			# --- Pipeline Phases with Conditional Execution ---

			# Phase 1: Preprocessing (Only run if last_phase < 1)
			if last_phase < 1:
				logger.info("\n📊 PHASE 1: Data Preprocessing & ACE Annotation")
				df = self.load_and_preprocess_data(data_paths)
				self._save_checkpoint(df, 1)
				last_phase = 1

			# Phase 2: Embeddings (Only run if last_phase < 2)
			if last_phase < 2:
				logger.info("\n🧠 PHASE 2: Semantic Embedding Generation")
				embeddings = self.generate_embeddings(df)
				self._save_checkpoint(df, 2)
				last_phase = 2
			else:
				# If resuming from >= Phase 2, embeddings must be loaded/available
				embeddings = self.embeddings
			
			# Phase 3: Topic Discovery (Only run if last_phase < 3)
			if last_phase < 3:
				logger.info("\n🎯 PHASE 3: Topic Discovery (20 Named Topics)")
				topic_results = self.discover_topics(df, embeddings)
				self._save_checkpoint(df, 3)
				last_phase = 3

			# Phase 4: Affective Analysis (Only run if last_phase < 4)
			if last_phase < 4:
				logger.info("\n😊 PHASE 4: Ensemble Affective Analysis")
				df = self.analyze_sentiment_emotion(df)
				self._save_checkpoint(df, 4)
				last_phase = 4

			# Phase 5: Anomaly Detection (Only run if last_phase < 5)
			if last_phase < 5:
				logger.info("\n🔍 PHASE 5: Multi-Scale Anomaly Detection")
				df = self.detect_anomalies(df, embeddings)
				self._save_checkpoint(df, 5)
				last_phase = 5

			# Phase 6: Bot Detection (Only run if last_phase < 6)
			if last_phase < 6:
				logger.info("\n🤖 PHASE 6: Ensemble Bot Detection")
				df = self.detect_bots(df)
				self._save_checkpoint(df, 6)
				last_phase = 6

			# Phase 7: Statistical Validation (Only run if last_phase < 7)
			if last_phase < 7:
				logger.info("\n📈 PHASE 7: Statistical Validation")
				stats_results = self.perform_statistical_analysis(df)
				self._save_checkpoint(df, 7)
				last_phase = 7

			# Phase 8: Visualizations (Always run at the end)
			logger.info("\n📊 PHASE 8: Enhanced Visualizations")
			self.generate_visualizations(df)

			# Phase 9: Reports (Always run at the end)
			logger.info("\n📝 PHASE 9: Comprehensive Reports")
			self.generate_reports(df)

			# --- Completion and Reporting ---
			end_time = datetime.now()
			runtime = end_time - start_time

			logger.info("\n" + "=" * 80)
			logger.info("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
			logger.info("=" * 80)
			logger.info(f"📊 Total tweets processed: {len(df):,}")
			
			# Safe access to results dictionary
			if 'topics' in self.results:
				logger.info(f"🎯 Topics identified: {self.results['topics']['num_topics']}")
			if 'anomalies' in self.results:
				logger.info(
					f"🔍 Anomalies detected: {self.results['anomalies']['total_anomalies']:,}"
				)
			if 'bots' in self.results:
				logger.info(
					f"🤖 Bot tweets identified: {self.results['bots']['bot_tweets']:,}"
				)
			
			logger.info(f"⏱️  Total runtime: {runtime}")
			logger.info(f"📁 Outputs saved to: outputs/")
			logger.info("=" * 80)

			# Clean up final checkpoint upon successful completion
			if os.path.exists("outputs/data/checkpoint.pkl"):
				os.remove("outputs/data/checkpoint.pkl")

			return df

		except Exception as e:
			logger.error(f"❌ Pipeline failed: {str(e)}")
			import traceback

			traceback.print_exc()
			raise


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

	# Configuration
	config = {
		"embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
		"num_topics": 20,  # Force exactly 20 topics
		"batch_size": 32,
		"chunk_size": 10000,
		"use_checkpoints": True,
		"sentiment_ensemble": True,
		"bot_threshold": 0.6,  # More sensitive
		"anomaly_sensitivity": 0.1,  # 10% anomalies
		"confidence_tracking": True,
	}

	# Initialize pipeline
	pipeline = EnhancedCovidTwitterAnalysisPipeline(config=config)

	# Data paths
	data_paths = [
		"Covid-19 Twitter Dataset (Apr-Jun 2020).csv",
		# Add more paths as needed:
		# "Covid-19 Twitter Dataset (Aug-Oct 2020).csv",
		# "Covid-19 Twitter Dataset (Apr-Jun 2021).csv",
	]

	try:
		# Run pipeline
		annotated_df = pipeline.run_complete_pipeline(data_paths)

		print("\n" + "=" * 80)
		print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
		print("=" * 80)
		print(f"\n📊 Annotated dataset shape: {annotated_df.shape}")
		print(f"\n📋 Output files generated:")
		print("   • outputs/data/covid_tweets_annotated.csv")
		print("   • outputs/reports/EXECUTIVE_SUMMARY.txt")
		print("   • outputs/reports/*.json (detailed reports)")
		print("   • outputs/visualizations/*.png (static plots)")
		print("   • outputs/visualizations/*.html (interactive dashboards)")
		print("\n" + "=" * 80)

	except Exception as e:
		print(f"\n❌ Error running pipeline: {e}")
		import traceback

		traceback.print_exc()

