# Baseline system G5

## Run
To run all sumulation 
```bash
cd NLP
python3 main.py
```
## Result from latest run 
```
1 epoch
Tagger accuracy for perceptron:
0.8732

Tagger accuracy for neural:
0.9075

Tagger accuracy for feature engenerd perceptron:
0.9172

Tagger accuracy for gold perceptron:
0.9382

UAS score for perceptron:
0.6659

UAS score for neural:
0.6900

--------
5 epochs
Taggers:
Tagger accuracy for perceptron:
0.8960

Tagger accuracy for neural:
0.9037

Tagger accuracy for feature engineerd perceptron:
0.9353

Tagger accuracy for gold perceptron:
0.9479

Parsers:
UAS score for perceptron:
0.6687

UAS score for neural:
0.7122

UAS score for feature engineered perceptron:
0.7520

```

## Structure

### constants/
containes all the global variabals 

### files/
containes all the taining and eval data

### lib/
containes all the classes for parser and tagger

### src/
containes all the scripts for training 

### util/
containes all the util functions like accuracy, UAS, make_vacabs ...

## TODO:

* read from stdin
* save output to file 
* save the traind network

```

 ****     ** **       ******* 
/**/**   /**/**      /**////**
/**//**  /**/**      /**   /**
/** //** /**/**      /******* 
/**  //**/**/**      /**////  
/**   //****/**      /**      
/**    //***/********/**      
//      /// //////// //       
```