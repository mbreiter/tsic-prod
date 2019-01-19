from django.urls import path, include
from django.conf.urls.static import static
from django.conf.urls import handler404, handler500
from django.conf import settings
from . import views

urlpatterns = [
    path('', views.home, name = 'landing'),
    path('home', views.home, name = 'home'),
    path('membership', views.membership, name = 'membership'),
    path('performance', views.performance, name = 'performance'),
    path('research', views.research, name = 'research'),
    path('register', views.register, name = 'register'),
    path('about', views.about, name = 'about'),
    path('about/edit', views.edit_about, name = 'edit_about'),
    path('tools', views.tools, name = 'tools'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

handler404 = 'website.views.error_404'
handler500 = 'website.views.error_500'
