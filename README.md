--------
WORK IN PROGRESS
--------
TODO: Refactor Code
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
Use [pretrained model](https://bitbucket.org/urmig/workspace/projects/BNEN) for the stacked bn-en parser
