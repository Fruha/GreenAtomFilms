from django.shortcuts import render, redirect
from django.views import View
from django.views.generic.list import ListView
from django.conf import settings
from django.http.request import HttpRequest
from filmsai import models
from filmsai import forms
from filmsai.utils import predict
from django.http import Http404
from django.utils.translation import gettext as _


class FilmsView(ListView):
    template_name = 'filmsai/pages/films_list.jinja2'
    model = models.Films
    paginate_by = settings.PAGINATION
    ordering = ['title']


class AddComment(View):
    template_name = 'filmsai/pages/add_comment.jinja2'
    form = forms.CommentCreateForm

    def get_context(self, film_id):
        return {'film':models.Films.objects.get(id=film_id)}

    def get(self, request:HttpRequest, film_id, **kwargs):
        form = self.form()
        context = self.get_context(film_id)
        context['form'] = form
        return render(request, self.template_name, context)

    def post(self, request:HttpRequest, film_id, **kwargs):
        data = request.POST.dict()
        data.update({'film': film_id})
        form = self.form(data)
        if form.is_valid():
            msg = form.save()
            predict(msg.id)
            return redirect('film_details', film_id=film_id)
        
        context = self.get_context(film_id)
        context['form'] = form
        return render(request, self.template_name, context)


class FilmDetailsView(ListView):
    template_name = 'filmsai/pages/films_details.jinja2'
    model = models.Comment
    paginate_by = settings.PAGINATION
    ordering = ['-datetime']
    
    def get_queryset(self, *args, **kwargs):
        self.queryset = models.Comment.objects.filter(film=args[0])
        return super().get_queryset()

    def get(self, request, film_id, *args, **kwargs):
        film = models.Films.objects.get(id=film_id)
        self.object_list = self.get_queryset(film)
        allow_empty = self.get_allow_empty()

        if not allow_empty:
            if self.get_paginate_by(self.object_list) is not None and hasattr(
                self.object_list, "exists"
            ):
                is_empty = not self.object_list.exists()
            else:
                is_empty = not self.object_list
            if is_empty:
                raise Http404(
                    _("Empty list and “%(class_name)s.allow_empty” is False.")
                    % {
                        "class_name": self.__class__.__name__,
                    }
                )
        context = self.get_context_data(*args,**kwargs)
        context['film'] = film
        return self.render_to_response(context)


def info(request):
    return render(request, "filmsai/pages/info.jinja2")