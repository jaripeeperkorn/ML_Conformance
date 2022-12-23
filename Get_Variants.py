def get_variants_list(lst): #get all of the variants in a list, return as list
    st = set(tuple(i) for i in lst) #convert list into set of tuples
    lst2 = list(st) #convert set of tuples into list of tuples
    return [list(e) for e in lst2]

def count_variant(log, variant): #count how many times a variant comes up in list
    c = 0
    for trace in log:
        if trace == variant:
            c += 1
    return(c)

def get_counts(log, variants):
    counts = []
    for var in variants:
        counts.append(count_variant(log, var))
    return counts

def get_variants_and_counts(log):
    variants = get_variants_list(log)
    return variants, get_counts(log, variants)
    

def get_variants_and_counts_and_wherezero(log, other_variants):
    variants = get_variants_list(log)
    places_where_zero = [] #add indcices for variants which also exist in other list
    for i in range(0, len(variants)):
        if variants[i] in other_variants:
            places_where_zero.append(i)
    return variants, get_counts(log, variants), places_where_zero



def get_where_zero(variants, other_variants):
    places_where_zero = [] #add indcices for variants which also exist in other list
    for i in range(0, len(variants)):
        if variants[i] in other_variants:
            places_where_zero.append(i)
    return places_where_zero

