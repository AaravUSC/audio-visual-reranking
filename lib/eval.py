
import jiwer
from torch import nn
from typing import List, Set, Tuple


class WordErrorRate(nn.Module):
    def __init__(self):
        super(WordErrorRate, self).__init__()

        self.truth_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])
        self.hypothesis_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])

    def forward(self,
                pred,
                actual):
        return jiwer.wer(
            actual,
            pred,
            truth_transform=self.truth_transformation,
            hypothesis_transform=self.hypothesis_transformation
        )


class RecoveryRate(nn.Module):
    def __init__(self,
                 separate_nouns: bool = False,
                 nouns: Set = None,
                 only_clear_preceding: bool = False):
        super(RecoveryRate, self).__init__()

        self.separate_nouns = separate_nouns
        if separate_nouns:
            self.nouns = [noun.lower() for noun in nouns]
        self.only_clear_preceding = only_clear_preceding

        self.truth_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])
        self.hypothesis_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])

    def forward(self,
                pred: List[str],
                actual: List[str],
                noise_indices: List[List],
                index_padding: int = 2) -> Tuple[float, float]:
        n_correct = 0
        n_noise = 0
        if self.separate_nouns:
            n_correct_nouns = 0
            n_noise_nouns = 0
        pred = self.hypothesis_transformation(pred)
        actual = self.truth_transformation(actual)

        for transcript_pred, transcript_actual, indices in zip(pred, actual, noise_indices):
            if self.only_clear_preceding:
                indices = [index for index in indices \
                           if ((index - 1) not in indices) and \
                                ((index - 2) not in indices) and \
                                ((index - 3) not in indices)]

            # Add to respective count variables
            if self.separate_nouns:
                n_noise_nouns += len([index for index in indices \
                                     if transcript_actual[min(len(transcript_actual) - 1, index)] in self.nouns])
                n_noise += len([index for index in indices \
                               if transcript_actual[min(len(transcript_actual) - 1, index)] not in self.nouns])
            else:
                n_noise += len(indices)
            for i_word in indices:
                k = 1
                target_str = transcript_actual[min(len(transcript_actual) - 1, i_word)]
                correct = i_word < len(
                    transcript_pred) and transcript_pred[i_word] == target_str

                # Give word index some buffer room
                while not correct and k <= index_padding and len(transcript_pred) > 0:
                    correct = transcript_pred[min(len(transcript_pred) - 1, max(0, i_word - k))] == target_str or \
                        transcript_pred[min(len(transcript_pred) - 1, i_word + k)] == target_str
                    k += 1

                # Increment number correct
                if correct:
                    if self.separate_nouns:
                        if target_str in self.nouns:
                            n_correct_nouns += 1
                        else:
                            n_correct += 1
                    else:
                        n_correct += 1

        if self.separate_nouns:
            rr_nouns = 1 if n_noise_nouns == 0 else n_correct_nouns / n_noise_nouns
            rr_else = 1 if n_noise == 0 else n_correct / n_noise
            return (rr_nouns, rr_else)
        else:
            return  1 if n_noise == 0 else n_correct / n_noise
        

class TranscriptAccuracy(nn.Module):
    def __init__(self):
        super(TranscriptAccuracy, self).__init__()

        self.truth_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])
        self.hypothesis_transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ])

    def forward(self,
                pred,
                actual):
        pred = self.hypothesis_transformation(pred)
        actual = self.truth_transformation(actual)

        count = 0
        for transcript_pred, transcript_actual in zip(pred, actual):
            if " ".join(transcript_pred) == " ".join(transcript_actual):
                count += 1
        
        return count / len(actual)
