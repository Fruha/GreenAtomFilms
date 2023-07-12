
from filmsai import models
from django.apps import apps



def predict(comment_id: int) -> None:
    """
    Функция для вызова моделей
    """
    # comment - объект комментрания в базе данных, который будет изменен

    comment = models.Comment.objects.get(id=comment_id)
    outputs = apps.get_app_config("filmsai").model.predict([comment.comment])
    print(outputs)
    # Код для обновления полей коментария
    import random
    comment.rating = max(min(round(outputs['marks'][0]),10),1)
    comment.assesment = bool(round(outputs['labels'][0]))
    comment.save()
    pass