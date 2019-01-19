from django import forms
from .models import User
from tsic.settings import USER_TYPE_KEYS

class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())
    confirm_password = forms.CharField(widget=forms.PasswordInput())
    verify_user_type = forms.CharField(widget=forms.PasswordInput())
    staff_status = False
    user_type = 0

    class Meta:
        model = User
        fields = ('username', 'password', 'confirm_password', 'email', 'user_type', 'verify_user_type')

    def clean(self):
        cleaned_data = super(UserForm, self).clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')
        user_type = cleaned_data.get('user_type')
        verify_user_type = cleaned_data.get('verify_user_type')

        self.user_type = user_type

        email = cleaned_data.get('email')
        username = cleaned_data.get('username')

        print(USER_TYPE_KEYS[user_type])

        if password != confirm_password:
            raise forms.ValidationError("password confirmation does not match.")

        if email and User.objects.filter(email=email).exclude(username=username).exists():
            raise forms.ValidationError("someone already is using that email address.")

        if verify_user_type != USER_TYPE_KEYS[user_type] and verify_user_type != USER_TYPE_KEYS[0]:
            raise forms.ValidationError("failed to verify the user role.")

        if verify_user_type == USER_TYPE_KEYS[0]:
            self.staff_status = True

class UserEditForm(forms.ModelForm):

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'username', 'email', 'email_preferences')
