"""
Emotion detection for assistant responses.
Maps sentiment to Web-Eye-Animation emotions.
"""

import re
from typing import Optional

# Available emotions from Web-Eye-Animation
AVAILABLE_EMOTIONS = [
    "joy", "sadness", "surprise", "anger", "fear", 
    "disgust", "confusion", "love", "sleepy", "excitement"
]


class EmotionDetector:
    """
    Detects emotions from text using sentiment analysis and keyword matching.
    Maps to Web-Eye-Animation compatible emotions.
    """
    
    def __init__(self):
        """Initialize emotion detector with keyword patterns."""
        # Positive emotion keywords
        self.positive_keywords = [
            "happy", "great", "awesome", "wonderful", "excellent", "fantastic",
            "amazing", "good", "nice", "love", "enjoy", "pleased", "delighted",
            "excited", "thrilled", "glad", "joy", "smile", "laugh", "fun"
        ]
        
        # Negative emotion keywords
        self.negative_keywords = [
            "sad", "sorry", "unfortunately", "bad", "terrible", "awful",
            "disappointed", "upset", "angry", "frustrated", "annoyed",
            "worried", "concerned", "problem", "issue", "difficult", "hard"
        ]
        
        # Surprise keywords
        self.surprise_keywords = [
            "wow", "amazing", "incredible", "unbelievable", "surprising",
            "unexpected", "shocking", "remarkable"
        ]
        
        # Anger keywords
        self.anger_keywords = [
            "angry", "frustrated", "annoyed", "irritated", "mad", "upset",
            "furious", "rage"
        ]
        
        # Confusion keywords
        self.confusion_keywords = [
            "confused", "unclear", "unsure", "don't know", "not sure",
            "uncertain", "puzzled", "bewildered"
        ]
    
    def detect_emotion(self, text: str) -> Optional[str]:
        """
        Detect emotion from text using keyword matching and sentiment analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Emotion string compatible with Web-Eye-Animation, or None for neutral
        """
        if not text or not text.strip():
            return None
        
        text_lower = text.lower()
        
        # Count keyword matches
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        surprise_count = sum(1 for word in self.surprise_keywords if word in text_lower)
        anger_count = sum(1 for word in self.anger_keywords if word in text_lower)
        confusion_count = sum(1 for word in self.confusion_keywords if word in text_lower)
        
        # Check for specific emotions first (more specific patterns)
        if confusion_count > 0:
            return "confusion"
        
        if anger_count > 0:
            return "anger"
        
        if surprise_count >= 2:  # Need multiple surprise words
            return "surprise"
        
        # Check for strong positive/negative sentiment
        if positive_count >= 3:
            # Very positive -> excitement
            if any(word in text_lower for word in ["excited", "thrilled", "amazing", "incredible"]):
                return "excitement"
            return "joy"
        
        if negative_count >= 2:
            return "sadness"
        
        # Moderate sentiment
        if positive_count >= 2:
            return "joy"
        
        if negative_count >= 1:
            return "sadness"
        
        # Check for love/affection
        if any(word in text_lower for word in ["love", "adore", "cherish", "dear"]):
            return "love"
        
        # Check for sleepiness
        if any(word in text_lower for word in ["tired", "sleepy", "exhausted", "rest"]):
            return "sleepy"
        
        # Default: no emotion (neutral)
        return None
    
    def map_to_eye_emotion(self, sentiment: str) -> Optional[str]:
        """
        Map sentiment string to Web-Eye-Animation emotion.
        
        Args:
            sentiment: Sentiment string (e.g., "positive", "negative")
            
        Returns:
            Emotion string or None
        """
        sentiment_lower = sentiment.lower()
        
        if sentiment_lower in ["positive", "very_positive"]:
            return "joy"
        elif sentiment_lower in ["negative", "very_negative"]:
            return "sadness"
        elif sentiment_lower == "surprise":
            return "surprise"
        elif sentiment_lower == "anger":
            return "anger"
        
        return None
    
    def get_emotion_from_response(self, response_text: str) -> Optional[str]:
        """
        Extract emotion from assistant response text.
        This is the main method to use for detecting emotions.
        
        Args:
            response_text: Full or partial response text from assistant
            
        Returns:
            Emotion string or None for neutral
        """
        return self.detect_emotion(response_text)


# Global instance
_emotion_detector = None

def get_emotion_detector() -> EmotionDetector:
    """Get or create global emotion detector instance."""
    global _emotion_detector
    if _emotion_detector is None:
        _emotion_detector = EmotionDetector()
    return _emotion_detector

