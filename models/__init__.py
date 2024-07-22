from .models_gen.gen_vlkt import build as build_gen
from .models_hoiclip.hoiclip import build as build_models_hoiclip
from .models_hoiclip.hoiclikp_topk import build as topk
from .models_cues.gen_cues import build as build_models_cues


def build_model(args):
    if args.model_name == "HOICLIP":
        return build_models_hoiclip(args)
    elif args.model_name == "GEN":
        return build_gen(args)
    elif args.model_name == "GEN_cues":
        return build_models_cues(args)
    elif args.model_name == "TOPK":
        return topk(args)

    raise ValueError(f'Model {args.model_name} not supported')
