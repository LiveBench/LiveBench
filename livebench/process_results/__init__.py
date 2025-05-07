from .util import AnswerEvaluator
from .coding.utils import LCB_generation_process_results, code_generation_process_results
from .data_analysis.cta.utils import cta_process_results
from .data_analysis.tablejoin.utils import tablejoin_process_results
from .data_analysis.tablereformat.utils import tablereformat_process_results
from .instruction_following.utils import instruction_following_process_results
from .writing.connections.utils import connections_process_results
from .writing.plot_unscrambling.utils import plot_unscrambling_process_results
from .writing.typos.utils import typos_process_results
from .math.amps_hard.utils import ampshard_process_results
from .math.math_comp.utils import mathcomp_process_results
from .math.olympiad.utils import olympiad_process_results
from .reasoning.spatial.utils import spatial_process_results
from .reasoning.web_of_lies.utils import weboflies_process_results
from .reasoning.zebra_puzzle.utils import zebra_puzzle_process_results
from typing import cast

from LiveBench.livebench.types import Question, \
    LCBQuestion, \
    BCBQuestion, \
    CTAQuestion, \
    TablejoinQuestion, \
    TableReformatQuestion, \
    InstructionFollowingQuestion, \
    ConnectionsQuestion, \
    PlotUnscramblingQuestion, \
    TyposQuestion, \
    AMPSHardQuestion, \
    MathCompQuestion, \
    OlympiadQuestion, \
    SpatialQuestion, \
    WebOfLiesQuestion, \
    ZebraPuzzleQuestion

def get_answer_evaluator(question: Question) -> AnswerEvaluator[Question]:
    if isinstance(question, LCBQuestion):
        return cast(AnswerEvaluator[Question], LCB_generation_process_results)
    elif isinstance(question, BCBQuestion):
        return cast(AnswerEvaluator[Question], code_generation_process_results)
    elif isinstance(question, CTAQuestion):
        return cast(AnswerEvaluator[Question], cta_process_results)
    elif isinstance(question, TablejoinQuestion):
        return cast(AnswerEvaluator[Question], tablejoin_process_results)
    elif isinstance(question, TableReformatQuestion):
        return cast(AnswerEvaluator[Question], tablereformat_process_results)
    elif isinstance(question, InstructionFollowingQuestion):
        return cast(AnswerEvaluator[Question], instruction_following_process_results)
    elif isinstance(question, ConnectionsQuestion):
        return cast(AnswerEvaluator[Question], connections_process_results)
    elif isinstance(question, PlotUnscramblingQuestion):
        return cast(AnswerEvaluator[Question], plot_unscrambling_process_results)
    elif isinstance(question, TyposQuestion):
        return cast(AnswerEvaluator[Question], typos_process_results)
    elif isinstance(question, AMPSHardQuestion):
        return cast(AnswerEvaluator[Question], ampshard_process_results)
    elif isinstance(question, MathCompQuestion):
        return cast(AnswerEvaluator[Question], mathcomp_process_results)
    elif isinstance(question, OlympiadQuestion):
        return cast(AnswerEvaluator[Question], olympiad_process_results)
    elif isinstance(question, SpatialQuestion):
        return cast(AnswerEvaluator[Question], spatial_process_results)
    elif isinstance(question, WebOfLiesQuestion):
        return cast(AnswerEvaluator[Question], weboflies_process_results)
    elif isinstance(question, ZebraPuzzleQuestion):
        return cast(AnswerEvaluator[Question], zebra_puzzle_process_results)
    else:
        raise ValueError(f"No answer evaluator found for question type: {type(question)}")
