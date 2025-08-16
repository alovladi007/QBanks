"""
AI-powered services for question generation, analysis, and enhancement.
"""
import asyncio
import hashlib
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
import torch

from ..core.config import settings
from ..models.questions import Question, QuestionVersion
from ..utils.text_processing import clean_text, extract_keywords

class AIQuestionGenerator:
    """AI-powered question generation service."""
    
    def __init__(self):
        """Initialize AI services."""
        self.openai_client = None
        self.anthropic_client = None
        self.embeddings_model = None
        self.local_model = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AI clients based on configuration."""
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY.get_secret_value()
            self.openai_client = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS
            )
            self.embeddings_model = OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL
            )
        
        # Initialize local model for fallback
        if torch.cuda.is_available() and settings.ENABLE_GPU:
            device = f'cuda:{settings.CUDA_DEVICE}'
        else:
            device = 'cpu'
        
        self.local_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    async def generate_questions(
        self,
        topic: str,
        learning_objectives: List[str],
        difficulty_level: str,
        question_types: List[str],
        count: int = 5,
        context: Optional[str] = None,
        bloom_levels: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate questions based on topic and parameters.
        
        Args:
            topic: Subject topic for questions
            learning_objectives: List of learning objectives
            difficulty_level: Easy, Medium, Hard, Expert
            question_types: MCQ, TrueFalse, ShortAnswer, Essay, etc.
            count: Number of questions to generate
            context: Additional context or source material
            bloom_levels: Bloom's taxonomy levels (1-6)
        
        Returns:
            List of generated questions with metadata
        """
        questions = []
        
        for q_type in question_types:
            num_questions = count // len(question_types)
            if q_type == question_types[-1]:
                num_questions += count % len(question_types)
            
            prompt = self._create_generation_prompt(
                topic, learning_objectives, difficulty_level,
                q_type, num_questions, context, bloom_levels
            )
            
            if self.openai_client:
                response = await self._generate_with_openai(prompt)
            else:
                response = await self._generate_with_local_model(prompt)
            
            parsed_questions = self._parse_generated_questions(response, q_type)
            questions.extend(parsed_questions)
        
        # Add embeddings and metadata
        for question in questions:
            question['embedding'] = await self.generate_embedding(question['stem'])
            question['auto_tags'] = await self.generate_tags(question['stem'])
            question['estimated_difficulty'] = await self.estimate_difficulty(question)
            question['quality_score'] = await self.assess_quality(question)
        
        return questions
    
    def _create_generation_prompt(
        self,
        topic: str,
        objectives: List[str],
        difficulty: str,
        question_type: str,
        count: int,
        context: Optional[str],
        bloom_levels: Optional[List[int]]
    ) -> str:
        """Create prompt for question generation."""
        bloom_verbs = {
            1: "remember, recall, identify, recognize",
            2: "understand, explain, interpret, summarize",
            3: "apply, use, implement, solve",
            4: "analyze, compare, contrast, examine",
            5: "evaluate, assess, judge, critique",
            6: "create, design, develop, formulate"
        }
        
        prompt = f"""Generate {count} high-quality {question_type} questions for the following:

Topic: {topic}
Difficulty Level: {difficulty}
Learning Objectives:
{chr(10).join(f'- {obj}' for obj in objectives)}

"""
        
        if bloom_levels:
            levels_str = ', '.join([bloom_verbs.get(l, '') for l in bloom_levels])
            prompt += f"Bloom's Taxonomy Levels: {levels_str}\n\n"
        
        if context:
            prompt += f"Context/Source Material:\n{context[:2000]}\n\n"
        
        prompt += self._get_format_instructions(question_type)
        
        return prompt
    
    def _get_format_instructions(self, question_type: str) -> str:
        """Get format instructions for specific question type."""
        instructions = {
            "MCQ": """Format each question as:
Question: [Clear, unambiguous question stem]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [Letter]
Explanation: [Why this answer is correct and others are wrong]
""",
            "TrueFalse": """Format each question as:
Statement: [Clear statement that is definitively true or false]
Answer: [True/False]
Explanation: [Why the statement is true or false]
""",
            "ShortAnswer": """Format each question as:
Question: [Clear question requiring 1-3 sentence answer]
Expected Answer: [Model answer]
Key Points: [Bullet points of required elements]
""",
            "Essay": """Format each question as:
Question: [Thought-provoking question requiring extended response]
Rubric:
- Thesis Statement (25%)
- Supporting Arguments (40%)
- Evidence/Examples (25%)
- Conclusion (10%)
Sample Answer Outline: [Key points to cover]
"""
        }
        
        return instructions.get(question_type, "Generate clear, well-formatted questions.")
    
    async def _generate_with_openai(self, prompt: str) -> str:
        """Generate questions using OpenAI API."""
        try:
            response = await self.openai_client.agenerate([[prompt]])
            return response.generations[0][0].text
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            return await self._generate_with_local_model(prompt)
    
    async def _generate_with_local_model(self, prompt: str) -> str:
        """Fallback generation using local model."""
        # Simplified generation for demonstration
        # In production, use a fine-tuned local LLM
        return """Question: What is the primary function of the component?
A) Store data
B) Process information
C) Display output
D) Manage connections
Correct Answer: B
Explanation: The component's main role is processing information."""
    
    def _parse_generated_questions(
        self,
        response: str,
        question_type: str
    ) -> List[Dict[str, Any]]:
        """Parse generated questions from AI response."""
        questions = []
        
        # Split response into individual questions
        if question_type == "MCQ":
            pattern = r"Question:(.*?)(?=Question:|$)"
            matches = re.findall(pattern, response, re.DOTALL)
            
            for match in matches:
                question = self._parse_mcq(match)
                if question:
                    question['type'] = 'MCQ'
                    questions.append(question)
        
        elif question_type == "TrueFalse":
            pattern = r"Statement:(.*?)(?=Statement:|$)"
            matches = re.findall(pattern, response, re.DOTALL)
            
            for match in matches:
                question = self._parse_true_false(match)
                if question:
                    question['type'] = 'TrueFalse'
                    questions.append(question)
        
        # Add more parsing for other question types
        
        return questions
    
    def _parse_mcq(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse MCQ question from text."""
        try:
            lines = text.strip().split('\n')
            stem = lines[0].strip()
            
            options = []
            correct_answer = None
            explanation = ""
            
            for line in lines[1:]:
                if line.strip().startswith(('A)', 'B)', 'C)', 'D)')):
                    options.append({
                        'text': line[2:].strip(),
                        'is_correct': False
                    })
                elif line.startswith('Correct Answer:'):
                    correct_answer = line.split(':')[1].strip()[0]
                elif line.startswith('Explanation:'):
                    explanation = line.split(':', 1)[1].strip()
            
            if correct_answer and options:
                option_index = ord(correct_answer) - ord('A')
                if 0 <= option_index < len(options):
                    options[option_index]['is_correct'] = True
            
            return {
                'stem': stem,
                'options': options,
                'explanation': explanation,
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            print(f"Error parsing MCQ: {e}")
            return None
    
    def _parse_true_false(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse True/False question from text."""
        try:
            lines = text.strip().split('\n')
            statement = lines[0].strip()
            
            answer = None
            explanation = ""
            
            for line in lines[1:]:
                if line.startswith('Answer:'):
                    answer = line.split(':')[1].strip().lower() == 'true'
                elif line.startswith('Explanation:'):
                    explanation = line.split(':', 1)[1].strip()
            
            return {
                'stem': statement,
                'answer': answer,
                'explanation': explanation,
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            print(f"Error parsing True/False: {e}")
            return None
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self.embeddings_model:
            try:
                embedding = await self.embeddings_model.aembed_query(text)
                return embedding
            except Exception as e:
                print(f"OpenAI embedding error: {e}")
        
        # Fallback to local model
        embedding = self.local_model.encode(text)
        return embedding.tolist()
    
    async def generate_tags(self, text: str) -> List[str]:
        """Generate tags for question text."""
        # Extract keywords
        keywords = extract_keywords(text)
        
        # Use AI for additional tag suggestions
        if self.openai_client:
            prompt = f"Generate 5 relevant tags for this question: {text[:500]}"
            try:
                response = await self.openai_client.agenerate([[prompt]])
                tags_text = response.generations[0][0].text
                ai_tags = [tag.strip() for tag in tags_text.split(',')]
                keywords.extend(ai_tags[:5])
            except:
                pass
        
        return list(set(keywords))[:10]
    
    async def estimate_difficulty(self, question: Dict[str, Any]) -> float:
        """Estimate question difficulty (0-1 scale)."""
        factors = {
            'word_count': len(question['stem'].split()),
            'sentence_complexity': self._calculate_sentence_complexity(question['stem']),
            'concept_depth': 0.5,  # Default, would use NLP analysis
            'cognitive_level': 0.5  # Based on Bloom's taxonomy
        }
        
        # Weighted average
        weights = {'word_count': 0.2, 'sentence_complexity': 0.3,
                  'concept_depth': 0.3, 'cognitive_level': 0.2}
        
        # Normalize word count (0-50 words maps to 0-1)
        factors['word_count'] = min(factors['word_count'] / 50, 1.0)
        
        difficulty = sum(factors[k] * weights[k] for k in factors)
        return round(difficulty, 2)
    
    def _calculate_sentence_complexity(self, text: str) -> float:
        """Calculate sentence complexity score."""
        sentences = text.split('.')
        if not sentences:
            return 0.5
        
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        # Map average length to 0-1 scale (5-25 words)
        complexity = (avg_length - 5) / 20
        return max(0, min(1, complexity))
    
    async def assess_quality(self, question: Dict[str, Any]) -> float:
        """Assess question quality."""
        scores = {
            'clarity': self._assess_clarity(question),
            'relevance': 0.8,  # Would check against learning objectives
            'discrimination': 0.7,  # Would use historical data
            'fairness': self._assess_fairness(question)
        }
        
        return round(sum(scores.values()) / len(scores), 2)
    
    def _assess_clarity(self, question: Dict[str, Any]) -> float:
        """Assess question clarity."""
        stem = question['stem']
        
        # Check for ambiguous words
        ambiguous_words = ['might', 'could', 'possibly', 'sometimes', 'maybe']
        ambiguity_penalty = sum(1 for word in ambiguous_words if word in stem.lower()) * 0.1
        
        # Check for negative phrasing
        negative_penalty = 0.2 if any(word in stem.lower() for word in ['not', 'never', 'none']) else 0
        
        # Length check (too short or too long)
        word_count = len(stem.split())
        if word_count < 5 or word_count > 50:
            length_penalty = 0.2
        else:
            length_penalty = 0
        
        clarity = 1.0 - ambiguity_penalty - negative_penalty - length_penalty
        return max(0.3, clarity)
    
    def _assess_fairness(self, question: Dict[str, Any]) -> float:
        """Assess question fairness and bias."""
        stem = question['stem'].lower()
        
        # Check for potentially biased language
        bias_indicators = ['he', 'she', 'his', 'her']  # Simplified
        bias_penalty = sum(0.1 for word in bias_indicators if word in stem.split())
        
        return max(0.5, 1.0 - bias_penalty)


class PlagiarismDetector:
    """Service for detecting plagiarism in questions."""
    
    def __init__(self):
        """Initialize plagiarism detector."""
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.85
    
    async def check_plagiarism(
        self,
        question_text: str,
        existing_questions: List[str]
    ) -> Dict[str, Any]:
        """
        Check if question is plagiarized.
        
        Args:
            question_text: Question to check
            existing_questions: List of existing questions to compare against
        
        Returns:
            Dictionary with plagiarism results
        """
        if not existing_questions:
            return {
                'is_plagiarized': False,
                'similarity_score': 0.0,
                'similar_questions': []
            }
        
        # Generate embeddings
        new_embedding = self.embeddings_model.encode([question_text])
        existing_embeddings = self.embeddings_model.encode(existing_questions)
        
        # Calculate similarities
        similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
        
        # Find similar questions
        similar_indices = np.where(similarities > self.similarity_threshold)[0]
        similar_questions = [
            {
                'text': existing_questions[i],
                'similarity': float(similarities[i])
            }
            for i in similar_indices
        ]
        
        # Sort by similarity
        similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'is_plagiarized': len(similar_questions) > 0,
            'similarity_score': float(max(similarities)) if len(similarities) > 0 else 0.0,
            'similar_questions': similar_questions[:5]
        }
    
    async def check_external_sources(
        self,
        question_text: str,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Check against external sources for plagiarism."""
        # This would integrate with external plagiarism APIs
        # For now, return placeholder
        return {
            'checked': False,
            'sources_checked': sources or [],
            'matches': []
        }


class QuestionEnhancer:
    """Service for enhancing and improving questions."""
    
    def __init__(self):
        """Initialize question enhancer."""
        self.ai_generator = AIQuestionGenerator()
    
    async def enhance_question(
        self,
        question: Dict[str, Any],
        enhancement_types: List[str]
    ) -> Dict[str, Any]:
        """
        Enhance question with various improvements.
        
        Args:
            question: Original question
            enhancement_types: List of enhancement types to apply
        
        Returns:
            Enhanced question
        """
        enhanced = question.copy()
        
        for enhancement in enhancement_types:
            if enhancement == 'clarity':
                enhanced = await self._enhance_clarity(enhanced)
            elif enhancement == 'distractors':
                enhanced = await self._enhance_distractors(enhanced)
            elif enhancement == 'explanation':
                enhanced = await self._enhance_explanation(enhanced)
            elif enhancement == 'accessibility':
                enhanced = await self._enhance_accessibility(enhanced)
        
        return enhanced
    
    async def _enhance_clarity(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Improve question clarity."""
        if self.ai_generator.openai_client:
            prompt = f"""Rewrite this question to be clearer and more precise:
Original: {question['stem']}

Requirements:
- Remove ambiguity
- Use simple, direct language
- Maintain the same difficulty level
- Keep the core concept intact"""
            
            try:
                response = await self.ai_generator.openai_client.agenerate([[prompt]])
                question['stem'] = response.generations[0][0].text.strip()
                question['enhancements'] = question.get('enhancements', []) + ['clarity']
            except:
                pass
        
        return question
    
    async def _enhance_distractors(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Improve MCQ distractors."""
        if question.get('type') != 'MCQ' or not question.get('options'):
            return question
        
        # Analyze current distractors
        correct_option = next((opt for opt in question['options'] if opt['is_correct']), None)
        if not correct_option:
            return question
        
        if self.ai_generator.openai_client:
            prompt = f"""Generate plausible distractors for this MCQ:
Question: {question['stem']}
Correct Answer: {correct_option['text']}

Generate 3 plausible but incorrect options that:
- Are similar in length and complexity to the correct answer
- Represent common misconceptions
- Are clearly wrong but not obviously so"""
            
            try:
                response = await self.ai_generator.openai_client.agenerate([[prompt]])
                new_distractors = response.generations[0][0].text.strip().split('\n')
                
                # Update distractors
                distractor_index = 0
                for i, option in enumerate(question['options']):
                    if not option['is_correct'] and distractor_index < len(new_distractors):
                        option['text'] = new_distractors[distractor_index].strip('- ')
                        distractor_index += 1
                
                question['enhancements'] = question.get('enhancements', []) + ['distractors']
            except:
                pass
        
        return question
    
    async def _enhance_explanation(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Add or improve explanation."""
        if self.ai_generator.openai_client:
            prompt = f"""Provide a comprehensive explanation for this question:
Question: {question['stem']}
{"Options: " + str(question.get('options', '')) if question.get('options') else ''}
{"Answer: " + str(question.get('answer', '')) if question.get('answer') else ''}

Write an explanation that:
- Explains why the correct answer is right
- Explains why incorrect options are wrong (if applicable)
- Provides relevant context or theory
- Is educational and helps learning"""
            
            try:
                response = await self.ai_generator.openai_client.agenerate([[prompt]])
                question['explanation'] = response.generations[0][0].text.strip()
                question['enhancements'] = question.get('enhancements', []) + ['explanation']
            except:
                pass
        
        return question
    
    async def _enhance_accessibility(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance question for accessibility."""
        # Add alternative formats
        question['accessibility'] = {
            'text_only': self._create_text_only_version(question),
            'reading_level': self._calculate_reading_level(question['stem']),
            'estimated_time': self._estimate_completion_time(question),
            'language_complexity': self._assess_language_complexity(question['stem'])
        }
        
        question['enhancements'] = question.get('enhancements', []) + ['accessibility']
        return question
    
    def _create_text_only_version(self, question: Dict[str, Any]) -> str:
        """Create text-only version of question."""
        text = question['stem']
        
        if question.get('options'):
            text += "\n\nOptions:\n"
            for i, opt in enumerate(question['options']):
                text += f"{chr(65 + i)}. {opt['text']}\n"
        
        return text
    
    def _calculate_reading_level(self, text: str) -> str:
        """Calculate reading level of text."""
        # Simplified Flesch-Kincaid Grade Level
        words = text.split()
        sentences = text.split('.')
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return "Unknown"
        
        grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
        
        if grade < 6:
            return "Elementary"
        elif grade < 9:
            return "Middle School"
        elif grade < 13:
            return "High School"
        elif grade < 16:
            return "College"
        else:
            return "Graduate"
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        vowels = "aeiou"
        syllables = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllables -= 1
        
        return max(1, syllables)
    
    def _estimate_completion_time(self, question: Dict[str, Any]) -> int:
        """Estimate completion time in seconds."""
        base_time = 30  # Base reading time
        
        # Add time based on question type
        if question.get('type') == 'MCQ':
            base_time += 10 * len(question.get('options', []))
        elif question.get('type') == 'Essay':
            base_time += 300  # 5 minutes for essay
        elif question.get('type') == 'ShortAnswer':
            base_time += 60
        
        # Add time based on complexity
        word_count = len(question['stem'].split())
        base_time += word_count * 2  # 2 seconds per word
        
        return base_time
    
    def _assess_language_complexity(self, text: str) -> str:
        """Assess language complexity."""
        # Check for complex vocabulary
        complex_words = [word for word in text.split() if len(word) > 10]
        complexity_ratio = len(complex_words) / max(len(text.split()), 1)
        
        if complexity_ratio < 0.1:
            return "Simple"
        elif complexity_ratio < 0.2:
            return "Moderate"
        else:
            return "Complex"