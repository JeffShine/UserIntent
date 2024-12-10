from model.infer_ocr import infer_ocr
from model.infer_cot import infer_cot

def build(config):
    
    if config['model']=='ocr':
       return infer_ocr(config['2b_path'],config['7b_path'])
    if config['model']=='cot':
        return infer_cot(config['2b_path'],config['7b_path'])
    