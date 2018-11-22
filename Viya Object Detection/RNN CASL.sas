proc cas;                                                /*2*/
	session mysess2;
	deepLearn.buildModel /                         
		modelTable={name="sosRnn", 
		replace=TRUE
		}
		type="RNN";
run;

deepLearn.modelInfo /                         
	modelTable={name="sosRnn"};
run;

deepLearn.addLayer /                   
	layer={type="INPUT"}               
	modelTable={name="sosRnn"}     
	name="data";
run;

deepLearn.addLayer /                   
	layer={type="recurrent"
	n=40
	act='sigmoid' 
	init='xavier' 
	rnnType='gru'
	outputType="samelength"
	}             
	modelTable={name="sosRnn"}      
	name="rnn1"
	srcLayers={"data"};
run;

deepLearn.addLayer / 
	layer={type="recurrent"
	n=40
	act="tanh"
	init="xavier"
	rnnType="GRU"
	outputType="encoding"
	}
	modelTable={name="sosRnn"}  
	name="rnn2"
	srcLayers={"rnn1"}
;
run;

deepLearn.addLayer /                   
	layer={type="output"
	act='exp' 
	init='xavier' 
	}         
	modelTable={name="sosRnn"}       
	name="outlayer"
	srcLayers={"rnn2"};
deepLearn.modelInfo /               
	modelTable={name="sosRnn"};
run;

table.fetch /                               
	table={name="sosRnn"};
run;

deepLearn.dlTrain /                                                     
	inputs={"Count_lag24","Count_lag1","Count_lag2"} 
	modelTable={name="sosRnn"}
	modelWeights={name="sosTrainedWeights",
	replace=TRUE
	}
	nThreads=1
	optimizer={algorithm={method="ADAM",
	lrPolicy='step', 
	gamma=0.5, 
	beta1=0.9, 
	beta2=0.999, 
	learningRate=0.0001
	},
	maxEpochs=20, 
	miniBatchSize=1
	}
	seed=54321
	table={name="training"}
	target="Count"
;
run;

deepLearn.dlScore /                                                       
	casOut={name="scored",
	replace=TRUE
	}
	copyvars={"Datetime","Count"}
	initWeights={name="sosTrainedWeights"}
	modelTable={name="sosRnn"}
	table={name="validation"}
;
run;

deepLearn.dlExportModel /                          
	casout={name="My_Model", replace=True}
	initWeights={name="sosTrainedWeights"}
	modelTable={name="sosRnn"};
run;

table.tableInfo /                                  
	name="My_Model";
run;

table.tableDetails /                               
	name="My_Model";
run;

quit;

data residuals;
	set mycas2.scored;
	Residual=Count-_DL_Pred_;
	AbsRes=abs(Residual);
	PredError=AbsRes/Count;
run;

proc sgplot data=residuals;
	series x=datetime y=count;
	series x=datetime y=_dl_pred_ / lineattrs=(pattern=dash);
run;

proc means data=residuals mean maxdec=2;
	var AbsRes PredError;
run;