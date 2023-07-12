from django.contrib import admin
from filmsai import models

@admin.register(models.Films)
class FilmsAdmin(admin.ModelAdmin):
    pass

@admin.register(models.Comment)
class CommentAdmin(admin.ModelAdmin):
    pass