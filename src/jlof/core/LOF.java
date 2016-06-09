/**
 * 
 */
package jlof.core;

import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.IntStream;


/**
 * @author Bostjan Kaluza
 * @date June 10, 2016
 */
 
 
public class LOF {
 
	/** The training instances  */
	private Collection<double[]> trainInstances;
	private int numAttributes, numInstances;
	
	/** The distances among instances. */
	private double [][] distTable;
 
	/** Indices of the sorted distance */
	private int [][] distSorted;
 
	/** The minimum values for training instances */
	private double [] minTrain;
 
	/** The maximum values training instances */
	private double [] maxTrain;
	
	/**
	 * @param trainCollection
	 */
	public LOF(Collection<double[]> trainCollection){
		
		// get training data dimensions
		numInstances = trainCollection.size();
		
		double[] first = trainCollection.iterator().next();
		numAttributes = first.length;
		
		trainInstances = trainCollection;
		
		// get the bounds for numeric attributes of training instances:
	    minTrain = new double[numAttributes];
	    maxTrain = new double [numAttributes];
	    
	    for (int i = 0; i < numAttributes; i++) {
	    	
	    	minTrain[i] = Double.POSITIVE_INFINITY;
	    	maxTrain[i] = Double.NEGATIVE_INFINITY;
	    	
	    	for(double[] instance : trainInstances){
	    		
	    		if(instance[i] < minTrain[i])
	    			minTrain[i] = instance[i];
	    		
	    		if(instance[i] > maxTrain[i])
	    			maxTrain[i] = instance[i];
	    	}
	    }
	    
		
		
		// fill the table with distances among training instances
	    distTable = new double[numInstances + 1][numInstances + 1];
		distSorted = new int[numInstances + 1][numInstances + 1];
		
	    int i = 0, j = 0;
		for(double[] instance1 :trainInstances){
			j = 0;
			for(double[] instance2 : trainInstances){
				distTable[i][j] = getDistance(instance1, instance2);
				j++;
			}
			i++;
		}
		
	}
	

	/**
	 * Returns neighbors for the new example.
	 * @param testInstance
	 * @param kNN
	 * @return
	 */
	public ArrayList<double[]> getNeighbors(double[] testInstance, int kNN){
		
		calcuateDistanceToTest(testInstance);
		
		// get the number of nearest neighbors for the current test instance:
		int numNN = getNNCount(kNN, numInstances);
		
		int[] nnIndex = new int[numNN];
		for (int i = 1; i <= numNN; i++) {
			nnIndex[i-1] = distSorted[numInstances][i];
	    }
		
		// loop over training data
		ArrayList<double[]> res = new ArrayList<double[]>(numNN);
		int idx = 0;
		for(double[] instance : trainInstances){
			// check if instance is among neighbors
			for(int j = 0; j < nnIndex.length; j++){
				if(nnIndex[j] == idx){
					res.add(instance);
					break;
				}
			}
			idx++;
		}

		return res;
	}
	
	/**
	 * Returns LOF score for new example.
	 * @param testInstance
	 * @param kNN
	 * @return
	 */
	public double getScore(double[] testInstance, int kNN){
		
		calcuateDistanceToTest(testInstance);
		
		return getLofIdx(numInstances, kNN);	
	}
	
	/**
	 * Returns LOF scores for training examples.
	 * @param kNN
	 * @return
	 */
	public double[] getTrainingScores(int kNN){
		double[] res = new double[numInstances];
		for(int idx = 0; idx < numInstances; idx++){
			res[idx] = getLofIdx(idx, kNN);
		}
		return res;
	}
	
	private double getLofIdx(int index, int kNN){
		// get the number of nearest neighbors for the current test instance:
		int numNN = getNNCount(kNN, index);

		// get LOF for the current test instance:
		double lof = 0.0;
		for (int i = 1; i <= numNN; i++) {
			double lrdi = getLocalReachDensity(kNN, index);
			lof += (lrdi == 0) ? 0 : getLocalReachDensity(kNN, distSorted[index][i]) / lrdi;
	    }
		lof /= numNN;

		return lof;	
	}
	
	private void calcuateDistanceToTest(double[] testInstance){
		// update the table with distances among training instances and the current test instance:
		int i = 0;
		for(double[] trainInstance : trainInstances){
			distTable[i][numInstances] = getDistance(trainInstance, testInstance);
			distTable[numInstances][i] = distTable[i][numInstances];
			i++;
		}
		
		// sort the distances
		for (i = 0; i < numInstances + 1; i++) {
			distSorted[i] = sortedIndices(distTable[i]);
	    }			
	}
 
	private double getDistance(double[] first, double[] second) {

		// calculate Euclidian distance
		double distance = 0;
 
		for (int i = 0; i < this.numAttributes; i++) {  
			distance += Math.abs(first[i] - second[i]) / (maxTrain[i] - minTrain[i]);
		}
 
		return distance;
	}
	
	private double getReachDistance(int kNN, int firstIndex, int secondIndex) {
		
		double reachDist = distTable[firstIndex][secondIndex];
		
		int numNN = getNNCount(kNN, secondIndex);
		
		if (distTable[secondIndex][distSorted[secondIndex][numNN]] > reachDist)
			reachDist = distTable[secondIndex][distSorted[secondIndex][numNN]];
		
		return reachDist;		
	}
	
	private int getNNCount(int kNN, int instIndex) {
 
		int numNN = kNN + 1;
		
		// if there are more neighbors with the same distance, take them too
		for (int i = kNN; i < this.numInstances - 1; i++) {
			if (distTable[instIndex][distSorted[instIndex][i]] == distTable[instIndex][distSorted[instIndex][i+1]])
				numNN++;
			else
				break;
		}
 
		return numNN;
	}
	
	private double getLocalReachDensity(int kNN, int instIndex) {
		
		// get the number of nearest neighbors:
		int numNN = getNNCount(kNN, instIndex);
 
		double lrd = 0;
 
		for (int i = 1; i < numNN; i++) {
			lrd += getReachDistance(kNN, instIndex, distSorted[instIndex][i]);
		}
		lrd = (lrd == 0) ? 0 : numNN / lrd;
 
		return lrd;
	}	
	    
	

	
	private int[] sortedIndices(double[] array){
		int[] sortedIndices = IntStream.range(0, array.length)
                .boxed().sorted((i, j) -> (int)(1000*(array[i] - array[j])))
                .mapToInt(ele -> ele).toArray();
		return sortedIndices;
	}
	
	
	public static void main(String[] args){
		
		int kNN = 5;
		
		ArrayList<double[]> data = new ArrayList<double[]>();
		data.add(new double[]{0, 0});
		data.add(new double[]{0, 1});
		data.add(new double[]{1, 0});
		data.add(new double[]{1, 1});
		data.add(new double[]{1, 2});
		data.add(new double[]{2, 1});
		data.add(new double[]{2, 2});
		data.add(new double[]{2, 0});
		data.add(new double[]{2, 0});
		data.add(new double[]{2, 0});
		data.add(new double[]{2, 0});
		
		LOF model = new LOF(data);
		System.out.println("LOF values on training examples");
		double[] scores = model.getTrainingScores(kNN);
		for(int i = 0; i < scores.length; i++){
			System.out.println(Arrays.toString(data.get(i)) + "\t" + scores[i]);
		}
		
		System.out.println("\nTest examples");
		double[] testSample = new double[]{2, 0};
		System.out.println(model.getScore(testSample, kNN));
		System.out.println(model.getScore(new double[]{0, 0}, kNN));
		System.out.println(model.getScore(new double[]{10, 4}, kNN));
		

		System.out.println("\nNeighbors of "+Arrays.toString(testSample));
		ArrayList<double[]> neighbors = model.getNeighbors(testSample, kNN);
		for(double[] n : neighbors){
			System.out.println(Arrays.toString(n));
		}
			
	}
	
}
