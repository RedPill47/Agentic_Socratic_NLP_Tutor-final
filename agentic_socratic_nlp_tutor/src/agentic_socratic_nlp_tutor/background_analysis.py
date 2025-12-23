"""
Background Multi-Agent System for Conversation Analysis

This MAS runs asynchronously using CrewAI agents to analyze conversations
and provide rich context to the fast-path SocraticTutor without blocking responses.

CrewAI Agents:
1. Learning Style Analyzer - Detects learning style from conversation patterns
2. Performance Evaluator - Assesses student responses and tracks trends
3. Knowledge Gap Detector - Identifies prerequisite gaps and misconceptions

Uses CrewAI for structured agent definitions and task management.
The extra 2-3 seconds overhead is acceptable since this runs in background.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import os
import json
import logging
import time

# Setup logging
logger = logging.getLogger(__name__)
# Don't set level here - let the root logger handle it
# This ensures logs appear in the console
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# CrewAI imports for proper agent system
try:
    from crewai import Agent, Task, Crew, LLM
    CREWAI_AVAILABLE = True
    logger.info("✅ CrewAI available - using CrewAI agents for background MAS")
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("⚠️ CrewAI not available - background MAS will use direct LLM calls")


@dataclass
class AnalysisResult:
    """Result from background MAS analysis"""
    learning_style: Optional[Dict[str, Any]] = None
    performance_assessment: Optional[Dict[str, Any]] = None
    knowledge_gaps: List[str] = field(default_factory=list)
    misconceptions: List[str] = field(default_factory=list)
    teaching_recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class LearningStyleAnalyzer:
    """CrewAI Agent that analyzes conversation patterns to detect learning style"""
    
    def __init__(self):
        if not CREWAI_AVAILABLE:
            # Fallback to direct LLM calls
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.use_crewai = False
        else:
            self.llm = LLM(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.agent = Agent(
                role="Learning Style Analyst",
                goal="Analyze student conversation patterns to identify their learning style",
                backstory="""You are an expert educational psychologist specializing in learning style detection.
                You analyze how students respond to questions, what language they use, and how they engage
                with different types of explanations to determine their preferred learning style.
                
                You look for indicators like:
                - Visual: "see", "show", "picture", "visualize", requests for diagrams
                - Auditory: "explain", "tell", "discuss", "sounds like"
                - Kinesthetic: "try", "practice", "hands-on", "example"
                - Reading: "documentation", "text", "written explanation"
                
                Provide your analysis in JSON format with primary_style, confidence, indicators, and recommendations.""",
                llm=self.llm,
                verbose=False,
                max_iter=1
            )
            self.use_crewai = True
    
    async def analyze(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze conversation to detect learning style using CrewAI agent"""
        start_time = time.time()
        logger.info("🔍 [LearningStyleAnalyzer] Starting analysis...")
        
        if len(conversation_history) < 3:
            logger.info("🔍 [LearningStyleAnalyzer] Skipping - not enough conversation history")
            return {"primary_style": "unknown", "confidence": 0.0}
        
        # Get recent exchanges (last 5-10 messages)
        recent = conversation_history[-10:]
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in recent
        ])
        logger.debug(f"🔍 [LearningStyleAnalyzer] Analyzing {len(recent)} messages")
        
        if self.use_crewai and CREWAI_AVAILABLE:
            # Use CrewAI agent
            try:
                logger.info("🔍 [LearningStyleAnalyzer] Using CrewAI agent...")
                task = Task(
                    description=f"""Analyze this conversation to detect the student's learning style.

Conversation:
{conversation_text}

Respond in JSON format:
{{
    "primary_style": "visual|auditory|kinesthetic|reading|unknown",
    "confidence": 0.0-1.0,
    "indicators": ["list of specific indicators found"],
    "recommendations": ["how to adapt teaching for this style"]
}}""",
                    agent=self.agent,
                    expected_output="JSON object with primary_style, confidence, indicators, and recommendations"
                )
                
                crew = Crew(
                    agents=[self.agent],
                    tasks=[task],
                    verbose=False
                )
                
                # Run in thread pool to avoid blocking
                logger.info("🔍 [LearningStyleAnalyzer] Running CrewAI crew...")
                result = await asyncio.to_thread(crew.kickoff)
                elapsed = time.time() - start_time
                logger.info(f"✅ [LearningStyleAnalyzer] Completed in {elapsed:.2f}s")
                
                # Parse result (CrewAI returns string, need to extract JSON)
                result_str = str(result)
                # Try to extract JSON from result
                import re
                json_match = re.search(r'\{[^{}]*\}', result_str, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    logger.info(f"✅ [LearningStyleAnalyzer] Detected style: {parsed.get('primary_style', 'unknown')}")
                    return parsed
                else:
                    # Fallback: try to parse entire result as JSON
                    parsed = json.loads(result_str)
                    logger.info(f"✅ [LearningStyleAnalyzer] Detected style: {parsed.get('primary_style', 'unknown')}")
                    return parsed
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ [LearningStyleAnalyzer] CrewAI error after {elapsed:.2f}s: {e}, falling back to direct LLM")
                # Fall through to direct LLM call
        else:
            # Fallback to direct LLM call
            logger.info("🔍 [LearningStyleAnalyzer] Using direct LLM call (CrewAI fallback)...")
            prompt = f"""Analyze this conversation to detect the student's learning style.

Conversation:
{conversation_text}

Identify indicators for:
- Visual: Uses words like "see", "show", "picture", "visualize", asks for diagrams
- Auditory: Uses words like "explain", "tell", "discuss", "sounds like"
- Kinesthetic: Uses words like "try", "practice", "hands-on", "example"
- Reading: Asks for "documentation", "text", "written explanation"

Respond in JSON format:
{{
    "primary_style": "visual|auditory|kinesthetic|reading|unknown",
    "confidence": 0.0-1.0,
    "indicators": ["list of specific indicators found"],
    "recommendations": ["how to adapt teaching for this style"]
}}"""

            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=300
                )
                
                result = json.loads(response.choices[0].message.content)
                elapsed = time.time() - start_time
                logger.info(f"✅ [LearningStyleAnalyzer] Completed in {elapsed:.2f}s (direct LLM)")
                logger.info(f"✅ [LearningStyleAnalyzer] Detected style: {result.get('primary_style', 'unknown')}")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ [LearningStyleAnalyzer] Error after {elapsed:.2f}s: {e}")
                return {"primary_style": "unknown", "confidence": 0.0}


class PerformanceEvaluator:
    """CrewAI Agent that evaluates student responses and tracks performance trends"""
    
    def __init__(self):
        if not CREWAI_AVAILABLE:
            # Fallback to direct LLM calls
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.use_crewai = False
        else:
            self.llm = LLM(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.agent = Agent(
                role="Response Evaluator",
                goal="Assess whether the student demonstrated understanding and track performance trends",
                backstory="""You are an expert educational assessor specializing in evaluating student responses
                in Socratic tutoring contexts. You analyze student answers to determine their level of understanding,
                looking for correct terminology, accurate explanations, logical reasoning, and evidence of learning progress.
                
                You provide numerical scores and detailed justification, tracking trends over time to identify
                whether the student is improving, declining, or maintaining stable performance.
                
                Provide your evaluation in JSON format with correctness, depth, score, trend, strengths, weaknesses, and recommendations.""",
                llm=self.llm,
                verbose=False,
                max_iter=1
            )
            self.use_crewai = True
    
    async def evaluate(
        self,
        question: str,
        student_response: str,
        context: str,
        previous_evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a single response using CrewAI agent"""
        start_time = time.time()
        logger.info("📊 [PerformanceEvaluator] Starting evaluation...")
        
        if self.use_crewai and CREWAI_AVAILABLE:
            # Use CrewAI agent
            try:
                task = Task(
                    description=f"""Evaluate this student response to a Socratic tutoring question.

Tutor's Question: {question}
Student's Response: {student_response}
Context: {context}

Previous Performance Trend:
{json.dumps(previous_evaluations[-5:] if previous_evaluations else [], indent=2)}

Evaluate:
1. Correctness (correct/partially_correct/incorrect)
2. Depth of understanding (shallow/moderate/deep)
3. Evidence of learning progress
4. Areas of confusion

Respond in JSON format:
{{
    "correctness": "correct|partially_correct|incorrect",
    "depth": "shallow|moderate|deep",
    "score": 0.0-1.0,
    "trend": "improving|declining|stable",
    "strengths": ["what they understood well"],
    "weaknesses": ["areas of confusion"],
    "recommendations": ["how to adjust teaching"]
}}""",
                    agent=self.agent,
                    expected_output="JSON object with correctness, depth, score, trend, strengths, weaknesses, and recommendations"
                )
                
                crew = Crew(
                    agents=[self.agent],
                    tasks=[task],
                    verbose=False
                )
                
                # Run in thread pool to avoid blocking
                logger.info("📊 [PerformanceEvaluator] Running CrewAI crew...")
                result = await asyncio.to_thread(crew.kickoff)
                elapsed = time.time() - start_time
                logger.info(f"✅ [PerformanceEvaluator] Completed in {elapsed:.2f}s")
                
                # Parse result (CrewAI returns string, need to extract JSON)
                result_str = str(result)
                import re
                json_match = re.search(r'\{[^{}]*\}', result_str, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    logger.info(f"✅ [PerformanceEvaluator] Score: {parsed.get('score', 0.0):.2f}, Trend: {parsed.get('trend', 'unknown')}")
                    return parsed
                else:
                    parsed = json.loads(result_str)
                    logger.info(f"✅ [PerformanceEvaluator] Score: {parsed.get('score', 0.0):.2f}, Trend: {parsed.get('trend', 'unknown')}")
                    return parsed
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ [PerformanceEvaluator] CrewAI error after {elapsed:.2f}s: {e}, falling back to direct LLM")
                # Fall through to direct LLM call
        else:
            # Fallback to direct LLM call
            logger.info("📊 [PerformanceEvaluator] Using direct LLM call (CrewAI fallback)...")
            prompt = f"""Evaluate this student response to a Socratic tutoring question.

Tutor's Question: {question}
Student's Response: {student_response}
Context: {context}

Previous Performance Trend:
{json.dumps(previous_evaluations[-5:] if previous_evaluations else [], indent=2)}

Evaluate:
1. Correctness (correct/partially_correct/incorrect)
2. Depth of understanding (shallow/moderate/deep)
3. Evidence of learning progress
4. Areas of confusion

Respond in JSON format:
{{
    "correctness": "correct|partially_correct|incorrect",
    "depth": "shallow|moderate|deep",
    "score": 0.0-1.0,
    "trend": "improving|declining|stable",
    "strengths": ["what they understood well"],
    "weaknesses": ["areas of confusion"],
    "recommendations": ["how to adjust teaching"]
}}"""

            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=400
                )
                
                result = json.loads(response.choices[0].message.content)
                elapsed = time.time() - start_time
                logger.info(f"✅ [PerformanceEvaluator] Completed in {elapsed:.2f}s (direct LLM)")
                logger.info(f"✅ [PerformanceEvaluator] Score: {result.get('score', 0.0):.2f}, Trend: {result.get('trend', 'unknown')}")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ [PerformanceEvaluator] Error after {elapsed:.2f}s: {e}")
                return {
                    "correctness": "unknown",
                    "depth": "unknown",
                    "score": 0.5,
                    "trend": "stable"
                }


class KnowledgeGapDetector:
    """CrewAI Agent that identifies prerequisite gaps and misconceptions"""
    
    def __init__(self, prerequisite_graph):
        self.prerequisite_graph = prerequisite_graph
        if not CREWAI_AVAILABLE:
            # Fallback to direct LLM calls
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.use_crewai = False
        else:
            self.llm = LLM(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.agent = Agent(
                role="Misconception Detector",
                goal="Identify misconceptions in student responses and detect missing prerequisite knowledge",
                backstory="""You are an expert educational diagnostician specializing in identifying student misconceptions
                and knowledge gaps. You detect factual errors, confused terminology, oversimplifications, and missing
                prerequisite knowledge that prevents students from fully understanding concepts.
                
                You are constructive, not critical - your goal is to help the tutor address these gaps effectively.
                You analyze student responses against prerequisite knowledge graphs to identify what foundational
                concepts are missing.
                
                Provide your analysis in JSON format with gaps, misconceptions, confusion_areas, and recommendations.""",
                llm=self.llm,
                verbose=False,
                max_iter=1
            )
            self.use_crewai = True
    
    async def detect_gaps(
        self,
        current_topic: str,
        student_responses: List[Dict[str, str]],
        mastered_concepts: List[str]
    ) -> Dict[str, Any]:
        """Detect knowledge gaps using CrewAI agent"""
        start_time = time.time()
        logger.info(f"🔎 [KnowledgeGapDetector] Starting gap detection for topic: {current_topic}")
        
        if not current_topic:
            logger.info("🔎 [KnowledgeGapDetector] Skipping - no current topic")
            return {"gaps": [], "misconceptions": []}
        
        # Get prerequisites from graph
        prerequisites = self.prerequisite_graph.get_all_prerequisites(current_topic)
        gaps = [p for p in prerequisites if p not in mastered_concepts]
        
        # Analyze responses for misconceptions
        recent_responses = "\n".join([
            f"Q: {r.get('question', '')}\nA: {r.get('answer', '')}"
            for r in student_responses[-5:]
        ])
        
        if self.use_crewai and CREWAI_AVAILABLE:
            # Use CrewAI agent
            try:
                task = Task(
                    description=f"""Analyze student responses to identify misconceptions about {current_topic}.

Student Responses:
{recent_responses}

Prerequisites for {current_topic}:
{', '.join(prerequisites)}

Mastered Concepts:
{', '.join(mastered_concepts) if mastered_concepts else 'None'}

Identify:
1. Specific misconceptions (incorrect beliefs)
2. Missing prerequisite knowledge
3. Confusion patterns

Respond in JSON format:
{{
    "gaps": ["list of missing prerequisites"],
    "misconceptions": ["specific incorrect beliefs"],
    "confusion_areas": ["topics causing confusion"],
    "recommendations": ["how to address gaps"]
}}""",
                    agent=self.agent,
                    expected_output="JSON object with gaps, misconceptions, confusion_areas, and recommendations"
                )
                
                crew = Crew(
                    agents=[self.agent],
                    tasks=[task],
                    verbose=False
                )
                
                # Run in thread pool to avoid blocking
                result = await asyncio.to_thread(crew.kickoff)
                
                # Parse result (CrewAI returns string, need to extract JSON)
                result_str = str(result)
                import re
                json_match = re.search(r'\{[^{}]*\}', result_str, re.DOTALL)
                if json_match:
                    parsed_result = json.loads(json_match.group())
                else:
                    parsed_result = json.loads(result_str)
                
                # Merge with graph-based gaps
                parsed_result["gaps"] = list(set(parsed_result.get("gaps", []) + gaps))
                elapsed = time.time() - start_time
                logger.info(f"✅ [KnowledgeGapDetector] Completed in {elapsed:.2f}s")
                logger.info(f"✅ [KnowledgeGapDetector] Found {len(parsed_result.get('gaps', []))} gaps, {len(parsed_result.get('misconceptions', []))} misconceptions")
                return parsed_result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ [KnowledgeGapDetector] CrewAI error after {elapsed:.2f}s: {e}, falling back to direct LLM")
                # Fall through to direct LLM call
        else:
            # Fallback to direct LLM call
            logger.info("🔎 [KnowledgeGapDetector] Using direct LLM call (CrewAI fallback)...")
            prompt = f"""Analyze student responses to identify misconceptions about {current_topic}.

Student Responses:
{recent_responses}

Prerequisites for {current_topic}:
{', '.join(prerequisites)}

Mastered Concepts:
{', '.join(mastered_concepts) if mastered_concepts else 'None'}

Identify:
1. Specific misconceptions (incorrect beliefs)
2. Missing prerequisite knowledge
3. Confusion patterns

Respond in JSON format:
{{
    "gaps": ["list of missing prerequisites"],
    "misconceptions": ["specific incorrect beliefs"],
    "confusion_areas": ["topics causing confusion"],
    "recommendations": ["how to address gaps"]
}}"""

            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=400
                )
                
                result = json.loads(response.choices[0].message.content)
                # Merge with graph-based gaps
                result["gaps"] = list(set(result.get("gaps", []) + gaps))
                elapsed = time.time() - start_time
                logger.info(f"✅ [KnowledgeGapDetector] Completed in {elapsed:.2f}s (direct LLM)")
                logger.info(f"✅ [KnowledgeGapDetector] Found {len(result.get('gaps', []))} gaps, {len(result.get('misconceptions', []))} misconceptions")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ [KnowledgeGapDetector] Error after {elapsed:.2f}s: {e}")
                return {"gaps": gaps, "misconceptions": []}


class BackgroundAnalysisMAS:
    """
    Multi-Agent System for background conversation analysis.
    
    Runs asynchronously to provide rich context to the fast-path tutor
    without blocking user responses.
    
    Analysis results are persisted to Supabase database for:
    - Survival across server restarts
    - Historical tracking
    - Cross-session learning
    """
    
    def __init__(self, prerequisite_graph, supabase_client=None):
        self.learning_style_analyzer = LearningStyleAnalyzer()
        self.performance_evaluator = PerformanceEvaluator()
        self.knowledge_gap_detector = KnowledgeGapDetector(prerequisite_graph)
        
        # Cache analysis results (in-memory for fast access)
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        
        # Supabase client for persistence (optional)
        self.supabase_client = supabase_client
        if self.supabase_client:
            logger.info("✅ [BackgroundMAS] Database persistence enabled")
        else:
            logger.warning("⚠️ [BackgroundMAS] Database persistence disabled - using in-memory cache only")
    
    async def analyze_conversation(
        self,
        session_id: str,
        conversation_history: List[Dict[str, str]],
        current_topic: Optional[str],
        mastered_concepts: List[str],
        last_question: Optional[str] = None,
        last_response: Optional[str] = None
    ) -> AnalysisResult:
        """
        Run comprehensive background analysis of the conversation.
        
        This runs asynchronously and doesn't block the response.
        Results are cached and used to enrich future responses.
        """
        start_time = time.time()
        logger.info(f"🚀 [BackgroundMAS] Starting analysis for session: {session_id[:20]}...")
        logger.info(f"🚀 [BackgroundMAS] Conversation history: {len(conversation_history)} messages")
        logger.info(f"🚀 [BackgroundMAS] Current topic: {current_topic or 'None'}")
        
        # Only analyze if we have enough conversation data
        if len(conversation_history) < 2:
            logger.info("🚀 [BackgroundMAS] Skipping - not enough conversation data")
            return AnalysisResult()
        
        # Run all analyses in parallel (non-blocking)
        tasks = []
        task_names = []
        
        # Learning style analysis (every 5 interactions)
        if len(conversation_history) % 5 == 0:
            logger.info("🚀 [BackgroundMAS] Adding LearningStyleAnalyzer task")
            tasks.append(
                self.learning_style_analyzer.analyze(conversation_history)
            )
            task_names.append("LearningStyleAnalyzer")
        else:
            logger.info("🚀 [BackgroundMAS] Skipping LearningStyleAnalyzer (not every 5th interaction)")
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Placeholder
            task_names.append("LearningStyleAnalyzer (skipped)")
        
        # Performance evaluation (if we have Q&A pair)
        if last_question and last_response:
            logger.info("🚀 [BackgroundMAS] Adding PerformanceEvaluator task")
            # Get previous evaluations from cache
            prev_eval = self.analysis_cache.get(session_id)
            prev_evaluations = []
            if prev_eval and prev_eval.performance_assessment:
                prev_evaluations = [prev_eval.performance_assessment]
            
            tasks.append(
                self.performance_evaluator.evaluate(
                    last_question,
                    last_response,
                    f"Topic: {current_topic or 'general'}",
                    prev_evaluations
                )
            )
            task_names.append("PerformanceEvaluator")
        else:
            logger.info("🚀 [BackgroundMAS] Skipping PerformanceEvaluator (no Q&A pair)")
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Placeholder
            task_names.append("PerformanceEvaluator (skipped)")
        
        # Knowledge gap detection
        student_responses = [
            {"question": msg.get("question", ""), "answer": msg.get("content", "")}
            for msg in conversation_history[-10:]
            if msg.get("role") == "user"
        ]
        logger.info("🚀 [BackgroundMAS] Adding KnowledgeGapDetector task")
        tasks.append(
            self.knowledge_gap_detector.detect_gaps(
                current_topic or "",
                student_responses,
                mastered_concepts
            )
        )
        task_names.append("KnowledgeGapDetector")
        
        logger.info(f"🚀 [BackgroundMAS] Running {len([t for t in task_names if 'skipped' not in t])} analysis tasks in parallel...")
        
        # Wait for all analyses (with timeout)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=10.0  # Increased timeout to 10 seconds
            )
            elapsed = time.time() - start_time
            logger.info(f"✅ [BackgroundMAS] All tasks completed in {elapsed:.2f}s")
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"⏱️ [BackgroundMAS] TIMEOUT after {elapsed:.2f}s (10s limit exceeded)")
            logger.warning(f"⏱️ [BackgroundMAS] This is acceptable - analysis runs in background")
            return AnalysisResult()
        
        # Extract results
        learning_style = results[0] if not isinstance(results[0], Exception) else None
        performance = results[1] if not isinstance(results[1], Exception) else None
        gap_analysis = results[2] if not isinstance(results[2], Exception) else {}
        
        # Log results
        if isinstance(results[0], Exception):
            logger.error(f"❌ [BackgroundMAS] LearningStyleAnalyzer failed: {results[0]}")
        if isinstance(results[1], Exception):
            logger.error(f"❌ [BackgroundMAS] PerformanceEvaluator failed: {results[1]}")
        if isinstance(results[2], Exception):
            logger.error(f"❌ [BackgroundMAS] KnowledgeGapDetector failed: {results[2]}")
        
        # Build comprehensive result
        analysis = AnalysisResult(
            learning_style=learning_style if learning_style else None,
            performance_assessment=performance if performance else None,
            knowledge_gaps=gap_analysis.get("gaps", []),
            misconceptions=gap_analysis.get("misconceptions", []),
            teaching_recommendations=(
                (learning_style.get("recommendations", []) if learning_style else []) +
                (performance.get("recommendations", []) if performance else []) +
                (gap_analysis.get("recommendations", []) if gap_analysis else [])
            ),
            confidence_score=(
                (learning_style.get("confidence", 0.0) if learning_style else 0.0) +
                (performance.get("score", 0.0) if performance else 0.0)
            ) / 2.0
        )
        
        # Cache result
        self.analysis_cache[session_id] = analysis
        
        total_elapsed = time.time() - start_time
        logger.info(f"✅ [BackgroundMAS] Analysis complete in {total_elapsed:.2f}s")
        logger.info(f"✅ [BackgroundMAS] Cached results for session: {session_id[:20]}...")
        
        return analysis
    
    def get_cached_analysis(self, session_id: str) -> Optional[AnalysisResult]:
        """Get cached analysis result for a session (from memory cache)"""
        return self.analysis_cache.get(session_id)
    
    async def load_from_database(self, session_db_id: str, user_id: str) -> Optional[AnalysisResult]:
        """
        Load latest analysis from database for a session.
        
        Args:
            session_db_id: Database UUID of the session
            user_id: User UUID
            
        Returns:
            Latest AnalysisResult or None if not found
        """
        if not self.supabase_client:
            return None
        
        try:
            # Get latest analysis for this session
            result = self.supabase_client.table('mas_analysis')\
                .select('*')\
                .eq('session_id', session_db_id)\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if result.data and len(result.data) > 0:
                data = result.data[0]
                analysis = AnalysisResult(
                    learning_style=data.get('learning_style'),
                    performance_assessment=data.get('performance_assessment'),
                    knowledge_gaps=data.get('knowledge_gaps', []),
                    misconceptions=data.get('misconceptions', []),
                    teaching_recommendations=data.get('teaching_recommendations', []),
                    confidence_score=data.get('confidence_score', 0.0),
                    timestamp=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')) if data.get('created_at') else datetime.now()
                )
                
                # Cache in memory for fast access
                # Use session_db_id as cache key (we'll need to map this)
                logger.info(f"✅ [BackgroundMAS] Loaded analysis from database for session: {session_db_id[:20]}...")
                return analysis
            
            return None
        except Exception as e:
            logger.error(f"❌ [BackgroundMAS] Failed to load from database: {e}")
            return None
    
    async def _save_to_database(
        self,
        session_id: str,
        analysis: AnalysisResult,
        session_db_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Save analysis result to database.
        
        Args:
            session_id: Frontend session ID (string)
            analysis: AnalysisResult to save
            session_db_id: Database UUID of the session (if available)
            user_id: User UUID (if available)
        """
        if not self.supabase_client:
            return
        
        # If session_db_id and user_id not provided, try to find them
        if not session_db_id or not user_id:
            # Try to get from sessions table using session_id
            try:
                session_result = self.supabase_client.table('sessions')\
                    .select('id, user_id')\
                    .eq('session_id', session_id)\
                    .limit(1)\
                    .execute()
                
                if session_result.data and len(session_result.data) > 0:
                    session_db_id = session_result.data[0]['id']
                    user_id = session_result.data[0]['user_id']
                else:
                    logger.warning(f"⚠️ [BackgroundMAS] Session not found in database: {session_id[:20]}...")
                    return
            except Exception as e:
                logger.error(f"❌ [BackgroundMAS] Failed to find session in database: {e}")
                return
        
        try:
            # Prepare data for database
            analysis_data = {
                'session_id': session_db_id,
                'user_id': user_id,
                'learning_style': analysis.learning_style,
                'performance_assessment': analysis.performance_assessment,
                'knowledge_gaps': analysis.knowledge_gaps,
                'misconceptions': analysis.misconceptions,
                'teaching_recommendations': analysis.teaching_recommendations,
                'confidence_score': analysis.confidence_score,
            }
            
            # Insert into mas_analysis table
            result = self.supabase_client.table('mas_analysis')\
                .insert(analysis_data)\
                .execute()
            
            # Update sessions table with latest analysis (for fast access)
            update_data = {
                'mas_analysis': {
                    'learning_style': analysis.learning_style,
                    'performance_assessment': analysis.performance_assessment,
                    'knowledge_gaps': analysis.knowledge_gaps,
                    'misconceptions': analysis.misconceptions,
                    'teaching_recommendations': analysis.teaching_recommendations,
                    'confidence_score': analysis.confidence_score,
                },
                'last_analysis_at': analysis.timestamp.isoformat()
            }
            
            self.supabase_client.table('sessions')\
                .update(update_data)\
                .eq('id', session_db_id)\
                .execute()
            
            # Update user-level learning style if detected
            if user_id and analysis.learning_style:
                try:
                    from agentic_socratic_nlp_tutor.user_profile_manager import UserProfileManager
                    user_profile_manager = UserProfileManager(self.supabase_client)
                    await user_profile_manager.update_learning_style(
                        user_id=user_id,
                        style=analysis.learning_style.get('primary_style'),
                        confidence=analysis.learning_style.get('confidence', 0.0)
                    )
                    logger.info(f"✅ [BackgroundMAS] Updated user-level learning style")
                except Exception as e:
                    logger.warning(f"⚠️ [BackgroundMAS] Error updating user-level learning style: {e}")
            
            logger.info(f"✅ [BackgroundMAS] Saved analysis to database for session: {session_db_id[:20]}...")
        except Exception as e:
            logger.error(f"❌ [BackgroundMAS] Failed to save analysis to database: {e}")
            raise
    
    def get_teaching_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get enriched teaching context from background analysis.
        
        This is called by the fast-path tutor to get additional context
        without blocking.
        """
        analysis = self.get_cached_analysis(session_id)
        
        if not analysis:
            return {}
        
        context = {}
        
        if analysis.learning_style:
            context["learning_style"] = analysis.learning_style.get("primary_style", "unknown")
            context["learning_style_confidence"] = analysis.learning_style.get("confidence", 0.0)
            context["learning_style_recommendations"] = analysis.learning_style.get("recommendations", [])
        
        if analysis.performance_assessment:
            context["performance_trend"] = analysis.performance_assessment.get("trend", "stable")
            context["performance_strengths"] = analysis.performance_assessment.get("strengths", [])
            context["performance_weaknesses"] = analysis.performance_assessment.get("weaknesses", [])
        
        if analysis.knowledge_gaps:
            context["knowledge_gaps"] = analysis.knowledge_gaps
            context["misconceptions"] = analysis.misconceptions
        
        if analysis.teaching_recommendations:
            context["teaching_recommendations"] = analysis.teaching_recommendations
        
        return context

