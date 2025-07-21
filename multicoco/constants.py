from enum import Enum
from typing import Dict, List
END_LATENT_TOKEN = '<|end_latent|>'
IMAGE_TOKEN = '<img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'  # Fixed: This should be the actual IMG_CONTEXT token
LATENT_TOKEN = '<|latent|>'
START_LATENT_TOKEN = '<|start_latent|>'
CHAT_START_TOKEN = '<|im_start|>'
CHAT_END_TOKEN = '<|im_end|>'
COCONUT_SPECIAL_TOKENS = [START_LATENT_TOKEN, LATENT_TOKEN, END_LATENT_TOKEN]
PROMPT_TOKENS = [CHAT_START_TOKEN, CHAT_END_TOKEN, IMAGE_TOKEN]

class AnswerChoice(Enum):
    A = '0'
    B = '1'
    C = '2'
    D = '3'
CHOICE_MAPPINGS: Dict[str, str] = {'a': AnswerChoice.A.value, 'first': AnswerChoice.A.value, 'zero': AnswerChoice.A.value, 'zeros': AnswerChoice.A.value, 'b': AnswerChoice.B.value, 'one': AnswerChoice.B.value, 'ones': AnswerChoice.B.value, 'second': AnswerChoice.B.value, 'c': AnswerChoice.C.value, 'third': AnswerChoice.C.value, 'two': AnswerChoice.C.value, 'twos': AnswerChoice.C.value, 'd': AnswerChoice.D.value, 'fourth': AnswerChoice.D.value, 'three': AnswerChoice.D.value, 'threes': AnswerChoice.D.value}
VALID_CHOICE_NUMBERS: List[str] = [choice.value for choice in AnswerChoice]
DEFAULT_DTYPE = 'bfloat16'
DEFAULT_MODEL_NAME = 'OpenGVLab/InternVL3-1B-Pretrained'
DEFAULT_MAX_LENGTH = 768
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 2
DEFAULT_EVAL_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-05
DEFAULT_NUM_EPOCHS = 3
DEFAULT_C_THOUGHT = 0
DEFAULT_MAX_LATENT_STAGE = 0
FALLBACK_IMAGE_SIZE = 448
LOSS_IGNORE_INDEX = -100
TEST_DATASET_LIMIT = 20
DEFAULT_LOG_DIR = 'logs'
DEFAULT_OUTPUT_DIR = 'checkpoints'
DEFAULT_EVAL_LOG_FORMAT = 'console'