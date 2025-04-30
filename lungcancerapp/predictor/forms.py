from django import forms

class PredictionForm(forms.Form):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]
    
    YES_NO_CHOICES = [
        (1, 'No'),
        (2, 'Yes'),
    ]
    
    gender = forms.ChoiceField(
        choices=GENDER_CHOICES, 
        widget=forms.RadioSelect,
        label="Gender"
    )
    
    age = forms.IntegerField(
        min_value=1, 
        max_value=120, 
        label="Age",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    smoking = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you smoke?"
    )
    
    yellow_fingers = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you have yellow fingers?"
    )
    
    anxiety = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you experience anxiety?"
    )
    
    peer_pressure = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you experience peer pressure?"
    )
    
    chronic_disease = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you have any chronic disease?"
    )
    
    fatigue = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you experience fatigue?"
    )
    
    allergy = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you have allergies?"
    )
    
    wheezing = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you experience wheezing?"
    )
    
    alcohol_consuming = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you consume alcohol?"
    )
    
    coughing = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you have a cough?"
    )
    
    shortness_of_breath = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you experience shortness of breath?"
    )
    
    swallowing_difficulty = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you have difficulty swallowing?"
    )
    
    chest_pain = forms.ChoiceField(
        choices=YES_NO_CHOICES, 
        widget=forms.RadioSelect,
        label="Do you experience chest pain?"
    )