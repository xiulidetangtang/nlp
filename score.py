import torch
import torch.nn as nn
import transformers
import io
import numpy as np

from data.preprocess import preprocess_text
from models.model import BaseModel
from utils.utils import loadfrom


def score(text : str, model : BaseModel, pretrained = None) -> dict:
    """
    :param model: trained model to score the text
    :param pretrained: pre trained tokenizer
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    text = preprocess_text(text)
    text = text.replace('\n', '[BR]')

    if pretrained == None:
        pretrained = 'microsoft/deberta-v3-base'

    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained,use_fast=False)
    tokens = tokenizer.tokenize(text)
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens = torch.tensor(tokens , dtype=torch.int).view(1,-1)
    mask = torch.tensor([1]*len(tokens) , dtype=torch.int).view(1,-1)

    model = model.to(device)
    tokens = tokens.to(device)
    mask = mask.to(device)

    scores = model(tokens, mask).detach().cpu().numpy()
    scores = scores[0]

    return {
        "cohesion"    : scores[0],
        "syntax"      : scores[1],
        "vocabulary"  : scores[2],
        "phraseology" : scores[3],
        "grammar"     : scores[4],
        "conventions" : scores[5]
    }

def load_model(model_path = './saved/model.pt'):
    with open(model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    state_dict = torch.load(buffer, weights_only = True, map_location=torch.device('cpu'))
    model = BaseModel()
    model.load_state_dict(state_dict)

    return model

if __name__ == '__main__':
    text = """Dear Generic_Name,

It has come to my attention that schools have partnerships with companies that allow students to explore certain occupations trough interships and shadowing opportunities, to discover whether they are interested in pursuing that type of job. However I'm in favor of the new opportunities that the schools are offering to young teenagers.

Allowing students to explore certain ocupations would motivate students to decide what type of job they want to dedicate their life on.

Every student dreams to have a perfect job, were they can have a good enviorenment and do what they love to do, But for many other reasons; students do not have the same opportunities as other's. with the new chances that companies are offering to students, everyone has the choice to be what they want and be succesful in life.

Every student goes trough a hard process were they have to decide, what they want to become in life. as a result not everyone gets the job they wanted to, causing that this person feels miserable working in a place were they don't feel comfortable. many juniors and seniors are feeling the pressure deciding what type of job they want; However companies allowing students to discover what their passion is and what type of job they want to get,would increase the percentage of students achieving their goals and get the job they wanted to.

Not every student has the chance to get the job they want to. either way because they don't have the sources to reach their goals or they can't because they are not well stablish financially. for example, my dream is to become a doctor but no one of my family have gone to collage, they don't have any expierence nor the sources to help achive my dream. I believe i'm not the only one that is going trough this, schools partenerships with companies benefits everyone in many differente ways, causing that every student has the chance to succed in life.

Lastly, allowing students to have the will to explore new opportunities benefit those who are still deciding what they want to become in life, and those who don't have any chance on achieving their goals. it's very important that every teenager prepares to the real world, without having any doubts or risking themselves to be unsuccesful. companies allowing students to choose what they want to be in life can be one of the many reasons that our future can be secure.

Thank you for taking your time on this matter and allowing me to express my opinion and my greatfullness.

Sincerely,

Generic_Name"""
    model = load_model()
    print(score(text, model))
    