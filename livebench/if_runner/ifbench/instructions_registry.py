# Copyright 2025 Allen Institute for AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all instructions."""

from livebench.if_runner.ifbench import instructions


def check_conflicts(instruction_ids: list[str]) -> list[tuple[str, str]]:
    """Check for conflicting instructions in a list.
    
    Args:
        instruction_ids: List of instruction IDs (e.g., ["count:word_count_range", "format:newline"])
    
    Returns:
        List of 2-tuples containing pairs of conflicting instruction IDs that are both present
        in the input list.
    """
    # Hardcoded pairs of conflicting instructions
    conflicting_pairs = [
        # ratio:sentence_type requires 2:1 declarative:interrogative ratio
        # ratio:sentence_balance requires 1:1:1 ratio of all three types
        ("ratio:sentence_type", "ratio:sentence_balance"),
        
        # words:repeats limits word repetition (max 1-5 times)
        # count:keywords_multiple requires specific keywords 1, 2, 3, 5, 7 times
        ("words:repeats", "count:keywords_multiple"),
        ("words:repeats", "format:emoji"),
        
        # format:no_whitespace requires no spaces/newlines
        # format:newline requires each word on a new line
        ("format:no_whitespace", "format:newline"),
        ("format:no_whitespace", "format:line_indent"),
        ("format:no_whitespace", "format:list"),
        ("format:no_whitespace", "format:thesis"),
        ("format:no_whitespace", "format:sub-bullets"),
        ("format:no_whitespace", "format:no_bullets_bullets"),
        ("format:no_whitespace", "format:parentheses"),
        ("format:no_whitespace", "format:quotes"),
        ("format:no_whitespace", "format:output_template"),
        
        # format:newline (each word on new line) conflicts with sentence-based instructions
        ("format:newline", "words:last_first"),
        ("format:newline", "words:paragraph_last_first"),
        ("format:newline", "sentence:alliteration_increment"),
        ("format:newline", "sentence:increment"),
        ("format:newline", "sentence:keyword"),
        ("format:newline", "ratio:sentence_type"),
        ("format:newline", "ratio:sentence_balance"),
        ("format:newline", "ratio:sentence_words"),
        ("format:newline", "format:no_bullets_bullets"),


        ("words:alphabet", "count:words_japanese"),
        ("words:alphabet", "words:last_first"),
        ("count:words_japanese", "words:consonants"),
        ("words:alphabet", "sentence:alliteration_increment")
    ]
    
    # Convert to set for efficient lookup
    instruction_set = set(instruction_ids)
    
    # Find conflicts that exist in the input list
    conflicts: list[tuple[str, str]] = []
    for id1, id2 in conflicting_pairs:
        if id1 in instruction_set and id2 in instruction_set:
            conflicts.append((id1, id2))
    
    return conflicts


INSTRUCTION_DICT: dict[str, type[instructions.Instruction]] = {
    "count:word_count_range": instructions.WordCountRangeChecker,
    "count:unique_word_count" : instructions.UniqueWordCountChecker,
    "ratio:stop_words" : instructions.StopWordPercentageChecker,
    "ratio:sentence_type" : instructions.SentTypeRatioChecker,
    "ratio:sentence_balance" : instructions.SentBalanceChecker,
    "count:conjunctions" : instructions.ConjunctionCountChecker,
    "count:person_names" : instructions.PersonNameCountChecker,
    "ratio:overlap" : instructions.NGramOverlapChecker,
    "count:numbers" : instructions.NumbersCountChecker,
    "words:alphabet" : instructions.AlphabetLoopChecker,
    "words:vowel" : instructions.ThreeVowelChecker,
    "words:consonants" : instructions.ConsonantClusterChecker,
    "sentence:alliteration_increment" : instructions.IncrementingAlliterationChecker,
    "words:palindrome" : instructions.PalindromeChecker,
    "count:punctuation" : instructions.PunctuationCoverChecker,
    "format:parentheses" : instructions.NestedParenthesesChecker,
    "format:quotes" : instructions.NestedQuotesChecker,
    "words:prime_lengths" : instructions.PrimeLengthsChecker,
    "format:options" : instructions.OptionsResponseChecker,
    "format:newline" : instructions.NewLineWordsChecker,
    "format:emoji" : instructions.EmojiSentenceChecker,
    "ratio:sentence_words" : instructions.CharacterCountUniqueWordsChecker,
    "count:words_japanese" : instructions.NthWordJapaneseChecker,
    "words:start_verb" : instructions.StartWithVerbChecker,
    "words:repeats" : instructions.LimitedWordRepeatChecker,
    "sentence:keyword" : instructions.IncludeKeywordChecker,
    "count:pronouns" : instructions.PronounCountChecker,
    "words:odd_even_syllables" : instructions.AlternateParitySyllablesChecker,
    "words:last_first" : instructions.LastWordFirstNextChecker,
    "words:paragraph_last_first" : instructions.ParagraphLastFirstWordMatchChecker,
    "sentence:increment" : instructions.IncrementingWordCountChecker,
    "words:no_consecutive" : instructions.NoConsecutiveFirstLetterChecker,
    "format:line_indent" : instructions.IndentStairsChecker,
    "format:quote_unquote" : instructions.QuoteExplanationChecker,
    "format:list" : instructions.SpecialBulletPointsChecker,
    "format:thesis" : instructions.ItalicsThesisChecker,
    "format:sub-bullets" : instructions.SubBulletPointsChecker,
    "format:no_bullets_bullets" : instructions.SomeBulletPointsChecker,
    "custom:multiples" : instructions.PrintMultiplesChecker,
    "custom:mcq_count_length": instructions.MultipleChoiceQuestionsChecker,
    "custom:reverse_newline": instructions.ReverseNewlineChecker,
    "custom:word_reverse": instructions.WordReverseOrderChecker,
    "custom:character_reverse": instructions.CharacterReverseOrderChecker,
    "custom:sentence_alphabet": instructions.SentenceAlphabetChecker,
    "custom:european_capitals_sort": instructions.EuropeanCapitalsSortChecker,
    "custom:csv_city": instructions.CityCSVChecker,
    "custom:csv_special_character": instructions.SpecialCharacterCSVChecker,
    "custom:csv_quotes": instructions.QuotesCSVChecker,
    "custom:date_format_list": instructions.DateFormatListChecker,
    "count:keywords_multiple" : instructions.KeywordsMultipleChecker,
    "words:keywords_specific_position" : instructions.KeywordSpecificPositionChecker,
    "words:words_position" : instructions.WordsPositionChecker,
    "repeat:repeat_change" : instructions.RepeatChangeChecker,
    "repeat:repeat_simple" : instructions.RepeatSimpleChecker,
    "repeat:repeat_span" : instructions.RepeatSpanChecker,
    "format:title_case" : instructions.TitleCaseChecker,
    "format:output_template" : instructions.OutputTemplateChecker,
    "format:no_whitespace" : instructions.NoWhitespaceChecker,
}
