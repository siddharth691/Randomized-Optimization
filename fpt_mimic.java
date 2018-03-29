
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

public class fpt_mimic {

	private static DecimalFormat df = new DecimalFormat("0.000");

    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    
    public static void main(String[] args) throws IOException {
    	
    	int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        
        int samples[] = {10, 50, 100, 200, 500, 1000};
        int rand_samp[] = {5, 10, 20, 50, 100, 500};
        
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        double start, end, time;
        String list_eval_mimic[] = {"0","0","0","0","0","0"};
        String list_time_mimic[] = {"0","0","0","0","0","0"};
        int iters = 2000;
        for(int index = 0; index < samples.length; index++)
        {
        	System.out.println("Current index" + index);
        	start = System.nanoTime();
	        MIMIC mimic = new MIMIC(samples[index], rand_samp[index], pop);
	        FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iters);
	        fit.train();
	        end = System.nanoTime();
	        time = (end - start)/Math.pow(10,9);
	        list_eval_mimic[index] = Double.toString(ef.value(mimic.getOptimal()));
	        list_time_mimic[index] = Double.toString(time);
        }
        
      //Writing the results to file
        BufferedWriter br = new BufferedWriter(new FileWriter("./results3/" + "fpt_mimic.csv"));
        
        //String builder for training error
        StringBuilder sb = new StringBuilder();
        for (String element : list_eval_mimic) {
         sb.append(element);
         sb.append(",");
        }

        sb.append(System.getProperty("line.separator"));
        //String builder for training time
        for (String element : list_time_mimic) {
         sb.append(element);
         sb.append(",");
        }
        
        br.write(sb.toString());
        br.close();
    	
    }
}
