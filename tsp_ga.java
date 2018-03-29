
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

public class tsp_ga {

	private static DecimalFormat df = new DecimalFormat("0.000");

    /** The n value */
	private static final int N = 50;
    
    public static void main(String[] args) throws IOException {
    	
    	Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        
        double start, end, time;
        int mate[] = {5, 10, 25, 50, 100};
	    int toMutate[] = {5, 10, 25, 50, 100};
	    int population[] = {10, 20, 50, 100, 200};
        
        String list_eval_ga[] = {"0","0","0","0","0"};
        String list_time_ga[] = {"0","0","0","0","0"};
        
        int iters = 2000;
        for(int index = 0; index < population.length; index++)
        {
        	System.out.println("Current index" + index);
        	start = System.nanoTime();
        	StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(population[index], mate[index], toMutate[index], gap);
        	FixedIterationTrainer fit = new FixedIterationTrainer(ga, iters);
            fit.train();
	        end = System.nanoTime();
	        time = (end - start)/Math.pow(10,9);
	        list_eval_ga[index] = Double.toString(ef.value(ga.getOptimal()));
	        list_time_ga[index] = Double.toString(time);
        }
        
      //Writing the results to file
        BufferedWriter br = new BufferedWriter(new FileWriter("./results3/" + "tsp_ga.csv"));
        
        //String builder for training error
        StringBuilder sb = new StringBuilder();
        for (String element : list_eval_ga) {
         sb.append(element);
         sb.append(",");
        }

        sb.append(System.getProperty("line.separator"));
        //String builder for training time
        for (String element : list_time_ga) {
         sb.append(element);
         sb.append(",");
        }
        
        br.write(sb.toString());
        br.close();
    	
    }
}
