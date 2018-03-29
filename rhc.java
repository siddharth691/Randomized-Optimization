import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import opt.prob.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class rhc {
    private static Instance[] instances = initializeInstances();
    private static Instance[] test_instances = initializeTestInstances();
    
	private static int inputLayer = 5, hiddenLayer = 10, outputLayer = 7;
	private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
	
	private static ErrorMeasure measure = new SumOfSquaresError();
	
	private static DataSet set = new DataSet(instances);
	
	private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
	private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];
	
	private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
	private static String[] oaNames = {"RHC"};
	private static String results = "";
	
	private static DecimalFormat df = new DecimalFormat("0.000");
	
	public static void main(String[] args) throws IOException {
		
		
	    for(int i = 0; i < nnop.length; i++) {
	        networks[i] = factory.createClassificationNetwork(
	            new int[] {inputLayer, hiddenLayer, outputLayer});
	        nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
	    }
	    
	    
	    oa[0] = new RandomizedHillClimbing(nnop[0]);
	    
	    //optimization algorithm number 
	    int i = 0;
	    
        train(oa[i], networks[i], oaNames[i], inputLayer, hiddenLayer, outputLayer); //trainer.train();
        
    }
	
        private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int inputLayer, int hiddenLayer, int outputLayer) throws IOException {
            System.out.println("\nError results for " + oaName + "\n---------------------------");
            
            int trainingIterations = 2000;
            String list_train_error [] = {"0","0","0","0","0","0","0","0","0","0","0","0"};
            String list_test_error [] = {"0","0","0","0","0","0","0","0","0","0","0","0"};
            
            String list_train_time [] = {"0","0","0","0","0","0","0","0","0","0","0","0"};
            String list_test_time [] = {"0","0","0","0","0","0","0","0","0","0","0","0"};
            
            
            int tind =0;
            double train_error =0, start_train = System.nanoTime(), end_train, trainingTime;
            for(int i = 1; i <= trainingIterations; i++) {
                oa.train();
                
                
                //Training Error after every iterations
                double error = 0;
                for(int j = 0; j < instances.length; j++) {
                    network.setInputValues(instances[j].getData());
                    network.run();
                    
                    Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                    /**
                     * Find the label with maximum probability
                     */
                    double max = 0;
                    int max_ind = 0;
                    
                    
                    for(int ind=0; ind< 7; ind++) {
                    	if(example.getContinuous(ind)>max)
                    	{
                    		max = example.getContinuous(ind);
                    		max_ind = ind;
                    	}
                    }
                    
                    // Create a double array of length 7, all values are initialized to 0
                    double[] example_encoded = new double[7];

                    // Set the i'th index to 1.0
                    example_encoded[max_ind] = 1.0; 
                    
                    example.setLabel(new Instance(example_encoded));
                    error += measure.value(output, example);
                }

                train_error = error;
                
                if((i == 100) || (i == 200) || (i == 300) || (i == 400) || (i == 500) || (i==700) || (i==900) || (i==1000) || (i == 1200) || (i == 1500) || (i == 1700) || (i ==2000))
                {	
                	
                	System.out.println("Iteration" + i + "done");
                	                	
                	//Updating the train end time and training time
                	end_train = System.nanoTime();
                	trainingTime = (end_train - start_train);
                	trainingTime /= Math.pow(10,9);
                	list_train_time[tind] = df.format(trainingTime);
                			
                	//If the current iteration append the training error to training error list
                	list_train_error[tind] = df.format((train_error/4500.0)*100.0);
                	
                	//Calculating the test error
	                //Create a new dummy network for testing the accuracy at the particular iterations
	                BackPropagationNetwork dummyNetwork = new BackPropagationNetwork();
	                dummyNetwork = factory.createClassificationNetwork(
	                        new int[] {inputLayer, hiddenLayer, outputLayer});
	                double start_test = System.nanoTime(), end_test, testingTime, correct = 0, incorrect = 0;
	                Instance optimalInstance = oa.getOptimal();
	    	        dummyNetwork.setWeights(optimalInstance.getData());
	    	        
	    	        //TESTING ERROR
	    	        start_test = System.nanoTime();
	    	        double test_error = 0;
	    	        for(int j = 0; j < test_instances.length; j++) {
	    	            dummyNetwork.setInputValues(test_instances[j].getData());
	    	            dummyNetwork.run();
	
	    	            Instance actual = test_instances[j].getLabel();
	    	            Instance predicted = new Instance(dummyNetwork.getOutputValues());
	    	            /**
	    	             * Find the label with maximum probability
	    	             */
	    	            double max = 0;
	    	            int max_ind = 0;
	    	            int correct_ind = 0;
	    	            
	    	            for(int ind=0; ind< 7; ind++) {
	    	            	if(predicted.getContinuous(ind)>max)
	    	            	{
	    	            		max = predicted.getContinuous(ind);
	    	            		max_ind = ind;
	
	    	            	}
	    	            	if(actual.getContinuous(ind) == 1.0) {
	    	            		correct_ind = ind;
	    	            	}
	    	            }
	    	            
	    	            double[] predicted_encode = new double[7];
	
	    	            // Set the i'th index to 1.0
	    	            predicted_encode[max_ind] = 1.0;
	    	            predicted.setLabel(new Instance(predicted_encode));
	    	            
	    	            double trash = correct_ind == max_ind ? correct++ : incorrect++;
	    	            test_error += measure.value(actual, predicted);
	    	        }
	    	        
	    	        //Updating the test error to list of test errors
	    	        list_test_error[tind] = df.format((test_error/1500.0)*100.0);
	    	        
	    	        //Updating the testing time to the array
	    	        end_test = System.nanoTime();
	    	        testingTime = end_test - start_test;
	    	        testingTime /= Math.pow(10,9);
	    	        list_test_time[tind] = df.format(testingTime);
	    	        
	    	        
	    	      //Updating the index
                	tind+=1;
    		    
    		    
                }   
    		    
    		    
            }
            
            
          //Writing the results to file
	        BufferedWriter br = new BufferedWriter(new FileWriter("./results3/rhc_results.csv"));
	        
	        //String builder for training error
	        StringBuilder sb = new StringBuilder();
	        for (String element : list_train_error) {
	         sb.append(element);
	         sb.append(",");
	        }

	        sb.append(System.getProperty("line.separator"));
	        //String builder for training time
	        for (String element : list_train_time) {
	         sb.append(element);
	         sb.append(",");
	        }

	        sb.append(System.getProperty("line.separator"));
	        //String builder for testing error
	        for (String element : list_test_error) {
	         sb.append(element);
	         sb.append(",");
	        }

	        sb.append(System.getProperty("line.separator"));
	        //String builder for testing time
	        for (String element : list_test_time) {
	         sb.append(element);
	         sb.append(",");
	        }

	        br.write(sb.toString());
	        br.close();
        }
        
        
        
        
        
        
        
        
        private static Instance[] initializeInstances() {

            double[][][] attributes = new double[4500][][];

            try {
                BufferedReader br = new BufferedReader(new FileReader(new File("src/cov_type_data.txt")));

                for(int i = 0; i < attributes.length; i++) {
                    Scanner scan = new Scanner(br.readLine());
                    scan.useDelimiter(",");

                    attributes[i] = new double[2][];
                    attributes[i][0] = new double[5]; // 7 attributes
                    attributes[i][1] = new double[1];

                    for(int j = 0; j < 5; j++)
                        attributes[i][0][j] = Double.parseDouble(scan.next());

                    attributes[i][1][0] = Double.parseDouble(scan.next());
                }
            }
            catch(Exception e) {
                e.printStackTrace();
            }

            Instance[] instances = new Instance[attributes.length];

            for(int i = 0; i < instances.length; i++) {
                instances[i] = new Instance(attributes[i][0]);
                
                // Read the digit 0-9 from the attribute array that was read from the csv
                int c = (int) attributes[i][1][0];
                           
                
                // Create a double array of length 7, all values are initialized to 0
                double[] classes = new double[7];

                // Set the i'th index to 1.0
                classes[c-1] = 1.0;
                instances[i].setLabel(new Instance(classes));
                
            }

            return instances;
        }
        
        
        private static Instance[] initializeTestInstances() {

            double[][][] attributes = new double[1500][][];

            try {
                BufferedReader br = new BufferedReader(new FileReader(new File("src/test_cov_type_data.txt")));

                for(int i = 0; i < attributes.length; i++) {
                    Scanner scan = new Scanner(br.readLine());
                    scan.useDelimiter(",");

                    attributes[i] = new double[2][];
                    attributes[i][0] = new double[5]; // 7 attributes
                    attributes[i][1] = new double[1];

                    for(int j = 0; j < 5; j++)
                        attributes[i][0][j] = Double.parseDouble(scan.next());

                    attributes[i][1][0] = Double.parseDouble(scan.next());
                }
            }
            catch(Exception e) {
                e.printStackTrace();
            }

            Instance[] instances = new Instance[attributes.length];

            for(int i = 0; i < instances.length; i++) {
                instances[i] = new Instance(attributes[i][0]);
                
                // Read the digit 0-9 from the attribute array that was read from the csv
                int c = (int) attributes[i][1][0];
                           
                
                // Create a double array of length 7, all values are initialized to 0
                double[] classes = new double[7];

                // Set the i'th index to 1.0
                classes[c-1] = 1.0;
                instances[i].setLabel(new Instance(classes));
                
            }

            return instances;
        }
    
    
    
    
}

