from django.http import HttpResponse, HttpResponseBadRequest


def strip_non_model_fields(data, model):
    """
    Remove any additional fields (i.e. those added onto a serializer) to format data for a model
    :param data:
    :param model:
    :return:
    """
    return dict([(key, val) for key, val in data.items() if key in [f.name for f in model._meta.get_fields()]])

