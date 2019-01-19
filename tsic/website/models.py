from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):

    USER_TYPE_CHOICES = (
        (1, 'analyst'),
        (2, 'associate'),
        (3, 'quant'),
    )

    EMAIL_PREFERENCES = (
        (0, 'get all updates on a daily basis'),
        (1, 'only for big changes in the portfolio'),
        (2, 'at the end of the semester'),
        (3, 'never'),
    )

    user_type = models.PositiveSmallIntegerField(choices=USER_TYPE_CHOICES,null=True, blank=True)
    email_preferences = models.PositiveSmallIntegerField(choices=EMAIL_PREFERENCES,null=True, blank=True, default=0)

class Analyst(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)

class Associate(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)

class Quant(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)

class ReportCategory(models.Model):
    name =  models.CharField(max_length=32)
    tagline = models.CharField(max_length=255, default="")

    class Meta:
        verbose_name_plural = "Categories"

    def __str__(self):
        return "%s" % self.name

    def __unicode__(self):
        return self.name

class Report(models.Model):
    id = models.AutoField(primary_key=True)
    authors = models.ManyToManyField(User, blank=True)
    category = models.ForeignKey(ReportCategory, on_delete=models.CASCADE)
    title = models.CharField(max_length=64, unique=True)
    sub_title = models.CharField(max_length=255, blank=True)
    published_on = models.DateField(auto_now=True)
    updated_on = models.DateField(auto_now=True)

    def generate_filename(self, filename):
        url = "%s/%s" % (self.category.name, filename)
        return url

    file = models.FileField(upload_to=generate_filename)

    class Meta:
        verbose_name_plural = "Reports"

    def __str__(self):
        return "%s" % self.title

    def __unicode__(self):
        return self.title
