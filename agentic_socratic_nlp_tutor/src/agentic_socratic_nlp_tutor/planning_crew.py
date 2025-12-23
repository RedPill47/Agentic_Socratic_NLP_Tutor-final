"""
On-Demand Planning MAS

Multi-agent system for creating personalized learning curricula.
Triggered when users explicitly request a learning plan.

Expected latency: 15-45 seconds (user expects this delay)
"""

import os
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# CrewAI imports
try:
    from crewai import Agent, Task, Crew, LLM
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("⚠️ CrewAI not available - Planning MAS will use direct LLM calls")


@dataclass
class LearningPlan:
    """Structured learning plan result."""
    goal: str
    target_concepts: List[str]
    knowledge_gaps: List[str]
    learning_path: List[Dict[str, Any]]  # Modules with concepts, time, resources
    total_time_estimate: int  # minutes
    learning_style_adaptations: Dict[str, Any]
    success_criteria: List[str]
    formatted_plan: str  # Final formatted markdown/text


class PlanningCrew:
    """
    Multi-agent system for creating personalized learning plans.
    
    Uses 5 CrewAI agents:
    1. Goal Analyzer - Parse learning objectives
    2. Gap Assessor - Identify knowledge gaps
    3. Path Builder - Create optimal learning sequence
    4. Resource Matcher - Match resources to modules
    5. Plan Formatter - Create final formatted document
    """
    
    def __init__(
        self,
        prerequisites,
        rag,
        learning_style: Optional[str] = None,
        mastered_concepts: Optional[List[str]] = None
    ):
        """
        Initialize Planning Crew.
        
        Args:
            prerequisites: PrerequisiteGraph instance
            rag: FastRAG instance
            learning_style: Student's learning style (if known)
            mastered_concepts: List of concepts student has mastered
        """
        self.prerequisites = prerequisites
        self.rag = rag
        self.learning_style = learning_style or "balanced"
        self.mastered_concepts = mastered_concepts or []
        
        if not CREWAI_AVAILABLE:
            logger.warning("⚠️ CrewAI not available - using fallback LLM calls")
            self.use_crewai = False
            # Fallback LLM client
            from openai import AsyncOpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.llm_client = AsyncOpenAI(api_key=api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            self.use_crewai = True
            self.llm = LLM(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self._setup_agents()
    
    def _setup_agents(self):
        """Setup CrewAI agents for planning."""
        if not self.use_crewai:
            return
        
        # 1. Goal Analyzer Agent
        self.goal_analyzer = Agent(
            role="Learning Goal Analyst",
            goal="Parse and analyze learning objectives from user requests",
            backstory="""You are an expert educational consultant specializing in curriculum design.
            You analyze learning requests to identify:
            - Specific target concepts or skills
            - Scope (mini-course vs full curriculum)
            - Time constraints or preferences
            - Learning depth (overview vs mastery)
            
            Extract clear, actionable learning objectives from user requests.""",
            llm=self.llm,
            verbose=False,
            max_iter=2
        )
        
        # 2. Gap Assessor Agent
        self.gap_assessor = Agent(
            role="Knowledge Gap Assessor",
            goal="Identify all prerequisite gaps between current knowledge and learning goals",
            backstory="""You are an expert in educational assessment and prerequisite analysis.
            You compare a student's current knowledge with their learning goals to identify:
            - Missing prerequisite concepts
            - Critical vs optional gaps
            - Knowledge depth requirements
            
            Use prerequisite relationships to ensure all gaps are identified.""",
            llm=self.llm,
            verbose=False,
            max_iter=2
        )
        
        # 3. Path Builder Agent
        self.path_builder = Agent(
            role="Learning Path Architect",
            goal="Create optimal learning sequence respecting prerequisites",
            backstory="""You are an expert instructional designer specializing in learning sequences.
            You create learning paths that:
            - Respect prerequisite relationships
            - Group concepts into logical modules
            - Define clear milestones
            - Estimate realistic time requirements
            - Ensure smooth progression from basics to advanced
            
            Build paths that are both pedagogically sound and efficient.""",
            llm=self.llm,
            verbose=False,
            max_iter=3
        )
        
        # 4. Resource Matcher Agent
        self.resource_matcher = Agent(
            role="Educational Resource Specialist",
            goal="Match learning resources to each module based on learning style",
            backstory="""You are an expert in educational resource curation and learning style adaptation.
            You find and match resources that:
            - Align with each learning module
            - Match the student's learning style (visual, auditory, kinesthetic, reading)
            - Include varied formats (text, diagrams, exercises, code)
            - Are appropriate for the difficulty level
            
            Create diverse, engaging resource lists for each module.""",
            llm=self.llm,
            verbose=False,
            max_iter=2
        )
        
        # 5. Plan Formatter Agent
        self.plan_formatter = Agent(
            role="Curriculum Document Writer",
            goal="Create a well-formatted, comprehensive learning plan document",
            backstory="""You are an expert technical writer specializing in educational documents.
            You create learning plans that are:
            - Well-structured and easy to follow
            - Include clear time estimates
            - Define success criteria
            - Provide tips and recommendations
            - Motivating and encouraging
            
            Write in a clear, professional, yet friendly tone.""",
            llm=self.llm,
            verbose=False,
            max_iter=2
        )
    
    async def create_plan(
        self,
        goal_request: str,
        current_knowledge: Optional[List[str]] = None,
        learning_style: Optional[str] = None,
        difficulty_preference: Optional[str] = None
    ) -> LearningPlan:
        """
        Create a personalized learning plan.
        
        Args:
            goal_request: User's learning goal request
            current_knowledge: List of concepts student knows (defaults to self.mastered_concepts)
            learning_style: Learning style preference (defaults to self.learning_style)
            difficulty_preference: Preferred difficulty level
            
        Returns:
            LearningPlan object with structured plan
        """
        logger.info(f"📚 [PlanningCrew] Starting plan creation for: {goal_request[:50]}...")
        
        current_knowledge = current_knowledge or self.mastered_concepts
        learning_style = learning_style or self.learning_style
        
        if self.use_crewai and CREWAI_AVAILABLE:
            return await self._create_plan_with_crewai(
                goal_request, current_knowledge, learning_style, difficulty_preference
            )
        else:
            return await self._create_plan_fallback(
                goal_request, current_knowledge, learning_style, difficulty_preference
            )
    
    async def _create_plan_with_crewai(
        self,
        goal_request: str,
        current_knowledge: List[str],
        learning_style: str,
        difficulty_preference: Optional[str]
    ) -> LearningPlan:
        """Create plan using CrewAI agents."""
        start_time = asyncio.get_event_loop().time()
        
        # Task 1: Goal Analysis
        goal_task = Task(
            description=f"""Analyze this learning goal request: "{goal_request}"

Extract:
1. Target concepts or skills the student wants to learn
2. Scope (overview, deep dive, full mastery)
3. Any time constraints mentioned
4. Learning depth preference

Return JSON:
{{
    "target_concepts": ["concept1", "concept2", ...],
    "scope": "overview|deep_dive|mastery",
    "time_constraint": "hours|days|weeks" or null,
    "depth": "surface|moderate|deep"
}}""",
            agent=self.goal_analyzer,
            expected_output="JSON object with target_concepts, scope, time_constraint, and depth"
        )
        
        # Task 2: Gap Assessment
        gap_task = Task(
            description=f"""Identify knowledge gaps for these target concepts: {goal_request}

Current knowledge: {', '.join(current_knowledge) if current_knowledge else 'Beginner - no prior knowledge'}

For each target concept:
1. Get all prerequisites using the prerequisite graph
2. Compare with current knowledge
3. Identify missing prerequisites
4. Prioritize: critical vs optional gaps

Return JSON:
{{
    "gaps": ["prerequisite1", "prerequisite2", ...],
    "critical_gaps": ["critical1", ...],
    "optional_gaps": ["optional1", ...]
}}""",
            agent=self.gap_assessor,
            expected_output="JSON object with gaps, critical_gaps, and optional_gaps",
            context=[goal_task]
        )
        
        # Task 3: Path Building
        path_task = Task(
            description=f"""Create an optimal learning path for: {goal_request}

Requirements:
- Respect all prerequisite relationships
- Group concepts into logical modules (3-5 modules)
- Order modules from basics to advanced
- Estimate time per module (in minutes)
- Define milestones for each module

Current knowledge: {', '.join(current_knowledge) if current_knowledge else 'None'}
Knowledge gaps: {gap_task.expected_output}

Return JSON:
{{
    "modules": [
        {{
            "name": "Module 1: Foundations",
            "concepts": ["concept1", "concept2"],
            "time_minutes": 120,
            "milestone": "description of what student should achieve"
        }},
        ...
    ],
    "total_time_minutes": 600
}}""",
            agent=self.path_builder,
            expected_output="JSON object with modules array and total_time_minutes",
            context=[goal_task, gap_task]
        )
        
        # Task 4: Resource Matching
        resource_task = Task(
            description=f"""Match learning resources to each module in the learning path.

Learning style: {learning_style}
Difficulty preference: {difficulty_preference or 'intermediate'}

For each module:
- Find relevant resources from the course materials (use RAG)
- Match to learning style (visual, auditory, kinesthetic, reading)
- Include varied formats (text, diagrams, code examples, exercises)
- Ensure resources match difficulty level

Return JSON:
{{
    "module_resources": [
        {{
            "module_name": "Module 1",
            "resources": [
                {{"type": "text", "title": "...", "description": "..."}},
                {{"type": "visual", "title": "...", "description": "..."}},
                ...
            ]
        }},
        ...
    ]
}}""",
            agent=self.resource_matcher,
            expected_output="JSON object with module_resources array",
            context=[path_task]
        )
        
        # Task 5: Plan Formatting
        format_task = Task(
            description=f"""Create a comprehensive, well-formatted learning plan document.

Combine all previous analysis:
- Goal: {goal_request}
- Target concepts: {goal_task.expected_output}
- Knowledge gaps: {gap_task.expected_output}
- Learning path: {path_task.expected_output}
- Resources: {resource_task.expected_output}

Create a markdown document with:
1. Overview (goal, time estimate, prerequisites needed)
2. Learning path with modules (concepts, time, milestones)
3. Resources for each module
4. Success criteria
5. Tips and recommendations

Write in a clear, professional, encouraging tone.""",
            agent=self.plan_formatter,
            expected_output="Complete formatted learning plan in markdown",
            context=[goal_task, gap_task, path_task, resource_task]
        )
        
        # Create and run crew
        crew = Crew(
            agents=[
                self.goal_analyzer,
                self.gap_assessor,
                self.path_builder,
                self.resource_matcher,
                self.plan_formatter
            ],
            tasks=[goal_task, gap_task, path_task, resource_task, format_task],
            process="sequential",  # Sequential execution
            verbose=False
        )
        
        logger.info("🚀 [PlanningCrew] Running CrewAI crew...")
        try:
            result = await asyncio.to_thread(crew.kickoff)
        except Exception as e:
            logger.error(f"❌ [PlanningCrew] Crew execution failed: {e}", exc_info=True)
            raise
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(f"✅ [PlanningCrew] Plan created in {elapsed:.2f}s")
        
        # Parse results and create LearningPlan
        try:
            return self._parse_plan_result(str(result), goal_request)
        except Exception as e:
            logger.error(f"❌ [PlanningCrew] Failed to parse plan result: {e}", exc_info=True)
            # Return a basic plan with the raw result
            return LearningPlan(
                goal=goal_request,
                target_concepts=[],
                knowledge_gaps=[],
                learning_path=[],
                total_time_estimate=0,
                learning_style_adaptations={"style": learning_style},
                success_criteria=[],
                formatted_plan=str(result) if result else "Failed to generate plan. Please try again."
            )
    
    async def _create_plan_fallback(
        self,
        goal_request: str,
        current_knowledge: List[str],
        learning_style: str,
        difficulty_preference: Optional[str]
    ) -> LearningPlan:
        """Fallback: Create plan using direct LLM calls."""
        logger.info("⚠️ [PlanningCrew] Using fallback LLM approach")
        
        # Simplified single LLM call for fallback
        prompt = f"""Create a personalized learning plan for: {goal_request}

Current knowledge: {', '.join(current_knowledge) if current_knowledge else 'Beginner'}
Learning style: {learning_style}
Difficulty: {difficulty_preference or 'intermediate'}

Create a comprehensive learning plan with:
1. Target concepts
2. Knowledge gaps
3. Learning path (modules with concepts and time estimates)
4. Resources for each module
5. Success criteria

Format as markdown."""

        try:
            completion = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational consultant. Create detailed learning plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            formatted_plan = completion.choices[0].message.content
            
            # Create basic LearningPlan from fallback
            return LearningPlan(
                goal=goal_request,
                target_concepts=[],
                knowledge_gaps=[],
                learning_path=[],
                total_time_estimate=0,
                learning_style_adaptations={},
                success_criteria=[],
                formatted_plan=formatted_plan
            )
        except Exception as e:
            logger.error(f"❌ [PlanningCrew] Fallback plan creation failed: {e}")
            raise
    
    def _parse_plan_result(self, result_text: str, goal_request: str) -> LearningPlan:
        """Parse CrewAI result into LearningPlan object."""
        # Try to extract JSON from result
        import re
        
        # Look for JSON in the result
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                # Extract structured data if available
                target_concepts = data.get("target_concepts", [])
                gaps = data.get("gaps", [])
                modules = data.get("modules", [])
                total_time = data.get("total_time_minutes", 0)
            except:
                # If JSON parsing fails, use defaults
                target_concepts = []
                gaps = []
                modules = []
                total_time = 0
        else:
            target_concepts = []
            gaps = []
            modules = []
            total_time = 0
        
        return LearningPlan(
            goal=goal_request,
            target_concepts=target_concepts,
            knowledge_gaps=gaps,
            learning_path=modules,
            total_time_estimate=total_time,
            learning_style_adaptations={"style": self.learning_style},
            success_criteria=[],
            formatted_plan=result_text
        )

