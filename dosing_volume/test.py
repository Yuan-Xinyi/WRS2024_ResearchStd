import model_loader as ml
import config_file as conf
ml.TransformerModel(conf.model_path["vit_d405"],(120,120),8,61)