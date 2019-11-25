# Inputs Files Paths
PATH_HOLIDAYS_2018="./datasets/holidays_2018.csv"
PATH_CONVERTIONS="./datasets/conversiones.csv"

PATH_PAGE_VIEWS="./datasets/pageviews.csv"
PATH_PAGE_VIEWS2="./datasets/pageviews_complemento.csv"

PATH_CONTENT_CATEGORY="./datasets/CONTENT_CATEGORY.csv"
PATH_CONTENT_CATEGORY_TOP="./datasets/CONTENT_CATEGORY_TOP.csv"
PATH_CONTENT_CATEGORY_BOTTOM="./datasets/CONTENT_CATEGORY_BOTTOM.csv"

PATH_DEVICE_DATA="./datasets/device_data.csv"

PATH_SITE="./datasets/SITE_ID.csv"
PATH_PAGE="./datasets/PAGE.csv"

# Output Files Paths
PATH_DT_6MONTHS="./datasets/data_hist_6m.csv"
PATH_DT_4MONTHS="./datasets/data_hist_3m.csv"
PATH_DT_5MONTHS="./datasets/data_hist_5m.csv"
PATH_DT_4M_6M="./datasets/data_tend_3m_6m.csv"
PATH_DT_CONVERSIONS="./datasets/data_conversions.csv"
PATH_RESULTS="./results/"

# Quantity of features to keep per Time's windows
K_BEST_FEATURES=850

# Months to use in the training dataset
K_TRAINING_MONTHS=[4,5,6,7,8]
K_FESELECTION_MONTHS=[4,5,6,7,8,9]

# Max number of bin to use when normalice a column
K_MAX_BINS = 200