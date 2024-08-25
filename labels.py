# PRODUCTION CARBONNE
CARBONNE   = 0
DECARBONNE = 1
NORMALE    = 2
production = [ "Carbonnée", "Décarbonnée", "Normale" ]

# Les mois
AOUT      = 0
AVRIL     = 1
DECEMBRE  = 2
FEVRIER   = 3
JANVIER   = 4
JUILLET   = 5
JUIN      = 6
MAI       = 7
MARS      = 8
NOVEMBRE  = 9
OCTOBRE   = 10
SEPTEMBRE = 11
mois = [ "Août", "Avril", "Décembre", "Février", "Janvier", "Juillet", "Juin", "Mai", "Mars", "Novembre", "Octobre", "Septembre" ]

# Jours de la semaine
DIMANCHE  = 0
JEUDI     = 1
LUNDI     = 2
MARDI     = 3
MERCREDI  = 4
SAMEDI    = 5
VENDREDI  = 6
jour = [ "Dimanche", "Jeudi", "Lundi", "Mardi", "Mercredi", "Samedi", "Vendredi" ]

# Type de jour férié
ONZE_NOVEMBRE   = 0   # 11novembre
QUATORZE_JUILLET= 1   # 14juillet
QUINZE_AOUT     = 2   # 15aout
PREMIER_JANVIER = 3   # 1janvier
PREMIER_MAI     = 4   # 1mai
PREMIER_NOVEMBRE= 5   # 1novembre
NOEL            = 6   # 25decembre
HUIT_MAI        = 7   # 8mai
ASCENSION       = 8   # ascension
NON_FERIE       = 9   # non férié
PENTECOTE       = 10  # pentecote
ferie = [ "Armistice 14-18", "Fête nationale", "Assomption", "Nouvel an", "Fête du travail", "Toussaint", "Noêl", "Armistice 39-45", "Ascension", "Non férié", "Pentecôte"]

# Les étiquettes des différentes colonnes du tableau numpy :
MixProdElec       = 0
EmissionCO2       = 1
PositionDansAnnee = 2
Annee             = 3
Mois              = 4
DemiHeure         = 5
Jour              = 6
JourFerie         = 7
JourFerieType     = 8
VacancesZoneA     = 9
VacancesZoneB     = 10
VacancesZoneC     = 11
Temperature       = 12
Nebulosite        = 13
Humidite          = 14
VitesseVent       = 15
Precipitation     = 16
columns = [ "Type production électrique", "Emission CO2", "Position dans l'année", "Année", "Mois", "Demi-heure", "Jour", "Jour férié", "Type jour férié", 
            "Vacances zone A", "Vacances zone B", "Vacances zone C", "Température", "Nébulosité", "Humidité", "Vitesse du vent", "Précipitation"]
