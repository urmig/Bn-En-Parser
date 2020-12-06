--------
WORK IN PROGRESS
--------
TODO: Refactor Code
# New : Model Deployed !
Try out [Bn-En Dependency Parser](http://ec2-18-214-87-91.compute-1.amazonaws.com/) <br/>
Input format : Sentence with normalized words concatenated by language ids : 'bn' for Bengali, 'en' for English. <br/>
For eg : Fun_en demand_en oV_bn SunaweV_bn hayZa_bn mAJeV_bn mAJeV_bn. <br/>

# Summary
POS Tagger + Dependency Parser for Bengali-English Code-Mixed Tweets

# Dependencies
pip install -r requirements.txt

# Train Parser
COMING SOON

# Test Parser 
* <b>CONLL Annotated Test/Dev File</b> <br/>
python bn-en_stacked_jm_parser.py --load /path/to/saved-model --bcdev /path/to/dev-or-test-file

* <b>Raw Text File</b> <br/>
python bn-en_stacked_jm_parser.py --load /path/to/saved-model --test /path/to/raw-text-file <br/>
See example testfile.txt

# Pre-Trained Models
Use [pretrained model](https://bitbucket.org/urmig/bn-en-parser-models/downloads/) for the stacked bn-en parser
