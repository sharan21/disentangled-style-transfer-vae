mkdir data
mkdir ./data/yelp

wget https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz -O data/yelp/data.tgz
tar -xvzf data/yelp/data.tgz -C data/yelp
mv data/yelp/yelp_review_polarity_csv/* ./data/yelp
rm -r data/yelp/yelp_review_polarity_csv/
rm data/yelp/data.tgz

mv ./data/yelp/test.csv ./data/yelp/yelp.test.csv 
mv ./data/yelp/train.csv ./data/yelp/yelp.train.csv 