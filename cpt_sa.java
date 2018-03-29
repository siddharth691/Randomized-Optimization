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

public class cpt_sa {

	private static DecimalFormat df = new DecimalFormat("0.000");

	/** The n value */
    private static final int N = 60;
    /** The t value */
    private static final int T = N / 10;
    
    public static void main(String[] args) throws IOException {
    	
    	int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        
        
        double start, end, time;
        double cooling[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        
        String list_eval_sa[] = {"0","0","0","0","0","0","0","0","0"};
        String list_time_sa[] = {"0","0","0","0","0","0","0","0","0"};
        
        int iters = 20000;
        for(int index = 0; index < cooling.length; index++)
        {
        	System.out.println("Current index" + index);
        	start = System.nanoTime();
        	SimulatedAnnealing sa = new SimulatedAnnealing(1E11, cooling[index], hcp);
        	FixedIterationTrainer fit = new FixedIterationTrainer(sa, iters);
            fit.train();
	        end = System.nanoTime();
	        time = (end - start)/Math.pow(10,9);
	        list_eval_sa[index] = Double.toString(ef.value(sa.getOptimal()));
	        list_time_sa[index] = Double.toString(time);
        }
        
      //Writing the results to file
        BufferedWriter br = new BufferedWriter(new FileWriter("./results3/" + "cpt_sa.csv"));
        
        //String builder for training error
        StringBuilder sb = new StringBuilder();
        for (String element : list_eval_sa) {
         sb.append(element);
         sb.append(",");
        }

        sb.append(System.getProperty("line.separator"));
        //String builder for training time
        for (String element : list_time_sa) {
         sb.append(element);
         sb.append(",");
        }
        
        br.write(sb.toString());
        br.close();
    	
    }
}
