def get_model(model_name, args):
    name = model_name.lower()
    if name == 'macil':
        from methods.macil import Learner
        return Learner(args)
    elif name == 'baseline':
        from methods.macil_cvpr_baseline import Learner
        return Learner(args)
    elif name == 'resnet':
        from methods.macil_cvpr_resnet import Learner
        return Learner(args)
    elif name == 'cvpr':
        from methods.macil_cvpr import Learner
        return Learner(args)

    else:
        raise ValueError("wrong model name")