import pandas
a = pandas.read_csv('~/Downloads/assignments_from_pool_45475229__03-11-2022.tsv', sep='\t')
a.dropna(axis=0, how='all', inplace=True)
print(
    a[a['OUTPUT:estimation'] != 'exact']['INPUT:Request'].head(),
    a[a['OUTPUT:estimation'] != 'exact']['INPUT:BannerTitle'].head()
)
