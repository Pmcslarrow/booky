def bucketize_age(age: int):
    if age < 18:
        return "<18"
    elif age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 50:
        return "36-50"
    else:
        return"50+"

def bucketize_year_of_publication(year_of_publication: int):
    if year_of_publication < 1950:
        return "pre-1950"
    elif year_of_publication < 1980:
        return "1950-1979"
    elif year_of_publication < 2000:
        return "1980-1999"
    elif year_of_publication < 2010:
        return "2000-2009"
    elif year_of_publication < 2020:
        return "2010-2019"
    else:
        return "2020+"
    