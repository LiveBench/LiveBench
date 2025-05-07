from pydantic import BaseModel
from typing import Any, Literal

class QuestionBase(BaseModel):
    question_id: str
    livebench_release_date: str
    livebench_removal_date: str
    turns: list[str]
    citation: str | None = None

class LCBQuestionBase(QuestionBase):
    public_test_cases: str
    private_test_cases: str
    question_title: str
    original_json: dict[str, Any]
    category: Literal['coding']

class LCBGenerationQuestion(LCBQuestionBase):
    task: Literal['LCB_generation']

class LCBCompletionQuestion(LCBQuestionBase):
    partial_solution: str
    remainder: str
    task: Literal['LCB_completion']

LCBQuestion = LCBGenerationQuestion | LCBCompletionQuestion

class BCBQuestionBase(QuestionBase):
    tests: str
    code_prompt: str
    libs: list[str]
    version: str
    entry_point: str
    difficulty: str
    expected_time: float
    question_title: str
    ground_truth: str
    category: Literal['coding']

class BCBGenerationQuestion(BCBQuestionBase):
    task: Literal['BCB_generation']

class BCBCompletionQuestion(BCBQuestionBase):
    partial_solution: str
    remainder: str
    task: Literal['BCB_completion']

BCBQuestion = BCBGenerationQuestion | BCBCompletionQuestion

class CTAQuestion(QuestionBase):
    ground_truth: str
    task: str = 'CTA'
    category: str = 'data_analysis'

class TablejoinQuestion(QuestionBase):
    ground_truth: str | dict[str, Any]
    task: str = 'tablejoin'
    category: str = 'data_analysis'

class TableReformatQuestion(QuestionBase):
    ground_truth: str
    task: str = 'tablereformat'
    category: str = 'data_analysis'

class InstructionFollowingQuestion(QuestionBase):
    instrction_id_list: list[str]
    kwargs: list[dict[str, str | int]]
    task_prompt: str
    category: str = 'instruction_following'
    task: Literal['simplify', 'paraphrase', 'summarize', 'story_generation']

class ConnectionsQuestion(QuestionBase):
    ground_truth: str
    raw_id: int
    task: Literal['connections']
    category: Literal['language']

class PlotUnscramblingQuestion(QuestionBase):
    ground_truth: str
    movie_name: str
    release_date: str
    task: Literal['plot_unscrambling']
    category: Literal['language']

class TyposQuestion(QuestionBase):
    ground_truth: str
    task: Literal['typos']
    category: Literal['language']

class AMPSHardQuestion(QuestionBase):
    ground_truth: str
    task: Literal['AMPS_Hard']
    category: Literal['math']
    subtask: str

class MathCompQuestion(QuestionBase):
    ground_truth: str
    task: Literal['math_comp']
    category: Literal['math']
    subtask: str

class OlympiadQuestion(QuestionBase):
    ground_truth: str
    expressions: str
    task: Literal['olympiad']
    category: Literal['math']
    subtask: str
    hardness: float
    release_date: int

class SpatialQuestion(QuestionBase):
    ground_truth: str
    category: Literal['reasoning']
    task: Literal['spatial']
    
class WebOfLiesQuestion(QuestionBase):
    ground_truth: str
    category: Literal['reasoning']
    task: Literal['web_of_lies_v2', 'web_of_lies_v3']

class ZebraPuzzleQuestion(QuestionBase):
    ground_truth: str
    category: Literal['reasoning']
    task: Literal['zebra_puzzle']
    level: int | None = None
    

Question = LCBGenerationQuestion \
    | LCBCompletionQuestion \
    | BCBGenerationQuestion \
    | BCBCompletionQuestion \
    | CTAQuestion \
    | TablejoinQuestion \
    | TableReformatQuestion \
    | InstructionFollowingQuestion \
    | ConnectionsQuestion \
    | PlotUnscramblingQuestion \
    | TyposQuestion \
    | AMPSHardQuestion \
    | MathCompQuestion \
    | OlympiadQuestion \
    | SpatialQuestion \
    | WebOfLiesQuestion \
    | ZebraPuzzleQuestion
