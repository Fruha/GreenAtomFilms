from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from PIL import Image


class Films(models.Model):
    title = models.CharField(
        max_length=512
    )
    image = models.FileField(
    )

    def save(self, *args, **kwargs):
        super().save()
        img = Image.open(self.image.path)
        resize = (300,160)
        img2 = img.resize(resize)
        img2.save(self.image.path)
        img.close()


class Comment(models.Model):

    film = models.ForeignKey(
        Films,
        on_delete=models.CASCADE,
        related_name='comments'
    )

    datetime = models.DateTimeField(
        auto_now=True
    )

    username = models.CharField(
        max_length=64
    )

    comment = models.TextField(

    )

    rating = models.IntegerField(
        validators=[
            MinValueValidator(0),
            MaxValueValidator(10)
        ],
        default=None,
        null=True,
    )

    assesment = models.BooleanField(
        default=None,
        null=True
    )

    def __str__(self) -> str:
        return f"<Comment: {self.pk=} {self.assesment=} {self.rating=}>"
