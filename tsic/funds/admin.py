from django.contrib import admin
from .models import *

admin.site.register(Fund)
admin.site.register(Portfolio)
admin.site.register(PortfolioStatistics)
admin.site.register(Optimization)
admin.site.register(Rebalance)
admin.site.register(Asset)
admin.site.register(View)
admin.site.register(Weight)
