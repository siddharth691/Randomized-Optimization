
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;


/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class tsp {
	
	private static DecimalFormat df = new DecimalFormat("0.000");

    /** The n value */
    private static final int N = 50;

    
    public static void main(String[] args) throws IOException {
        
    	int iterations[] = {50, 100, 200, 300, 500, 700, 1000, 5000, 10000};
    	double start, end;
    	int tot_run = 5;
    	
    	String list_rhc_time[] = {"0","0","0","0","0","0","0" ,"0","0"};
    	String list_sa_time[] = {"0","0","0","0","0","0","0" ,"0","0"};
    	String list_ga_time[] = {"0","0","0","0","0","0","0" ,"0","0"};
    	String list_mimic_time[] = {"0","0","0","0","0","0","0" ,"0","0"};
    	
    	String list_rhc_eval[] = {"0","0","0","0","0","0","0" ,"0","0"};
    	String list_sa_eval[] = {"0","0","0","0","0","0","0" ,"0","0"};
    	String list_ga_eval[] = {"0","0","0","0","0","0","0" ,"0","0"};
    	String list_mimic_eval[] = {"0","0","0","0","0","0","0" ,"0","0"};
    	
    	int iter =0;
    	for(int iters: iterations)
    		
    	{
    		
    		System.out.println("Current iteration" + iters);
    		double tot_eval_rhc = 0;
    		double tot_eval_sa = 0;
    		double tot_eval_ga = 0;
    		double tot_eval_mimic = 0;
    		
    		double time_rhc = 0;
    		double time_sa = 0;
    		double time_ga = 0;
    		double time_mimic = 0;
    		double time = 0;
    		for (int currun= 0; currun < tot_run; currun++)
    		{
    			
    			System.out.println("Current run" + currun);
    			
    			Random random = new Random();
    	        // create the random points
    	        double[][] points = new double[N][2];
    	        for (int i = 0; i < points.length; i++) {
    	            points[i][0] = random.nextDouble();
    	            points[i][1] = random.nextDouble();   
    	        }
		        
    	        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
    	        Distribution odd = new DiscretePermutationDistribution(N);
    	        NeighborFunction nf = new SwapNeighbor();
    	        MutationFunction mf = new SwapMutation();
    	        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
    	        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
    	        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
		        
		        
		        //Randomized hill climbing
		        start = System.nanoTime();
		        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
		        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iters);
		        fit.train();
		        end = System.nanoTime();
		        time = (end - start)/Math.pow(10,9);
		        time_rhc = time_rhc +  time;
		        tot_eval_rhc = tot_eval_rhc + ef.value(rhc.getOptimal());
		        		        
		        
		        //Simulated Annealing
		        start = System.nanoTime();
		        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
		        fit = new FixedIterationTrainer(sa, iters);
		        fit.train();
		        end = System.nanoTime();
		        time = (end - start)/Math.pow(10,9);
		        time_sa = time_sa +  time;
		        tot_eval_sa = tot_eval_sa + ef.value(sa.getOptimal());
		        
		        
		        //Genetic Algorithm
		        start = System.nanoTime();
		        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
		        fit = new FixedIterationTrainer(ga, iters);
		        fit.train();
		        end = System.nanoTime();
		        time = (end - start)/Math.pow(10,9);
		        time_ga = time_ga +  time;
		        tot_eval_ga = tot_eval_ga + ef.value(ga.getOptimal());
		        
		        
		        //Mimic
		        start = System.nanoTime();
		        ef = new TravelingSalesmanSortEvaluationFunction(points);
		        int[] ranges = new int[N];
		        Arrays.fill(ranges, N);
		        odd = new  DiscreteUniformDistribution(ranges);
		        Distribution df = new DiscreteDependencyTree(.1, ranges); 
		        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
		        
		        MIMIC mimic = new MIMIC(200, 100, pop);
		        fit = new FixedIterationTrainer(mimic, iters);
		        fit.train();
		        end = System.nanoTime();
		        time = (end - start)/Math.pow(10, 9);
		        time_mimic = time_mimic +  time;
		        tot_eval_mimic = tot_eval_mimic + ef.value(mimic.getOptimal());
	        
    		}
    		
    		//Averaging the total optimal fitness values and total time for test runs
    		tot_eval_rhc /= tot_run;
    		tot_eval_sa /= tot_run;
    		tot_eval_ga /= tot_run;
    		tot_eval_mimic /= tot_run;
    		
    		time_rhc /= tot_run;
    		time_sa /= tot_run;
    		time_ga /= tot_run;
    		time_mimic /= tot_run;
    		
    		//Updating the values to array
    		list_rhc_time[iter] = df.format(time_rhc);
    		list_sa_time[iter] = df.format(time_sa);
    		list_ga_time[iter] = df.format(time_ga);
    		list_mimic_time[iter] = df.format(time_mimic);
    		
    		list_rhc_eval[iter] = df.format(tot_eval_rhc);
    		list_sa_eval[iter] = df.format(tot_eval_sa);
    		list_ga_eval[iter] = df.format(tot_eval_ga);
    		list_mimic_eval[iter] = df.format(tot_eval_mimic);
    		
    		iter+=1;
    }
    	
    	//Writing strings to files
    	
    	//Writing the results to file
        BufferedWriter br = new BufferedWriter(new FileWriter("./results3/" + "travelling_salesman_test.csv"));
        
        //String builder for training error
        StringBuilder sb = new StringBuilder();
        for (String element : list_rhc_eval) {
         sb.append(element);
         sb.append(",");
        }

        sb.append(System.getProperty("line.separator"));
        //String builder for training time
        for (String element : list_sa_eval) {
         sb.append(element);
         sb.append(",");
        }

        sb.append(System.getProperty("line.separator"));
        //String builder for testing error
        for (String element : list_ga_eval) {
         sb.append(element);
         sb.append(",");
        }

        sb.append(System.getProperty("line.separator"));
        //String builder for testing time
        for (String element : list_mimic_eval) {
         sb.append(element);
         sb.append(",");
        }
        
        sb.append(System.getProperty("line.separator"));
        //String builder for testing time
        for (String element : list_rhc_time) {
         sb.append(element);
         sb.append(",");
        }
        
        sb.append(System.getProperty("line.separator"));
        //String builder for testing time
        for (String element : list_sa_time) {
         sb.append(element);
         sb.append(",");
        }
        
        sb.append(System.getProperty("line.separator"));
        //String builder for testing time
        for (String element : list_ga_time) {
         sb.append(element);
         sb.append(",");
        }
        
        sb.append(System.getProperty("line.separator"));
        //String builder for testing time
        for (String element : list_mimic_time) {
         sb.append(element);
         sb.append(",");
        }

        br.write(sb.toString());
        br.close();
    }
}
