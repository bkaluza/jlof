/**
 * 
 */
package jlof.core;

import java.lang.Math;
import java.io.FileReader;
import java.io.FileWriter;

import weka.core.*;

/**
 * @author Bostjan
 * @date 6.9.2011
 */
 
 
public class LOF {
 
	/** The training instances plus the current test instance. */
	Instances m_Instances;
	
	/** The distances among current instances. */
	private double [][] m_DistTable;
 
	/** The sorted distances of current instances. */
	private int [][] m_Sorted;
 
	/** The minimum values for numeric attributes of training instances. */
	private double [] m_MinTrain;
 
	/** The maximum values for numeric attributes of training instances. */
	private double [] m_MaxTrain;
	
	/** 
	 * Get minimum and maximum value - bounds - for each numeric atribute. If atribute is not numeric, 
	 * set min/max value to -/+ infinity. Values are stored in global tables m_MinTrain and m_MaxTrain.
	 */
	private void getMinMax(Instances trainInst) {
 
	    m_MinTrain = new double [trainInst.numAttributes()];
	    m_MaxTrain = new double [trainInst.numAttributes()];
	    for (int i = 0; i < trainInst.numAttributes(); i++) {
	    	m_MinTrain[i] = Double.POSITIVE_INFINITY;
	    	m_MaxTrain[i] = Double.NEGATIVE_INFINITY;
    		if (trainInst.attribute(i).isNumeric()) {
		    	for (int j = 0; j < trainInst.numInstances(); j++) {
		    		if (trainInst.instance(j).value(i) < m_MinTrain[i])
		    			m_MinTrain[i] = trainInst.instance(j).value(i);
		    		if (trainInst.instance(j).value(i) > m_MaxTrain[i])
		    			m_MaxTrain[i] = trainInst.instance(j).value(i);
		    	}
		    	if (Utils.eq(m_MinTrain[i], m_MaxTrain[i])) {
		    	    m_MinTrain[i] = 0;
		    	    m_MaxTrain[i] = 1;		    		
		    	}
    		}
	    }
	}
 
	private double getDistance(Instance first, Instance second) {
 
		double diff, distance = 0;
 
		for (int i = 0; i < m_Instances.numAttributes(); i++) { 
			
			diff = 0;
 
			if (!first.isMissing(i) && 
				!second.isMissing(i) && 
				(first.value(i) != second.value(i))) {
 
				switch (m_Instances.attribute(i).type()) {
 
				case Attribute.STRING:
				case Attribute.DATE:
					diff = 1;
					break;
 
				case Attribute.NOMINAL:					
					// circular attributes:
					if (m_Instances.attribute(i).name().contains("CIRC")) {
						diff = Math.abs(first.value(i) - second.value(i));
						if (diff > Math.floor(m_Instances.attribute(i).numValues() / 2))
							diff = m_Instances.attribute(i).numValues() - diff;		
						diff = 2 * diff / m_Instances.attribute(i).numValues();					
					}
					else {
						diff = 1;
					}
					break;
 
				case Attribute.NUMERIC:
					diff = Math.abs(first.value(i) - second.value(i));
					diff = diff / (m_MaxTrain[i] - m_MinTrain[i]);
					break;
				}
			}
			
			distance += diff;
		}
 
		return distance;
	}
	
	private double getReachDist(int kNN, int firstIndex, int secondIndex) {
		
		double reachDist = m_DistTable[firstIndex][secondIndex];
		
		int numNN = getNN(kNN, secondIndex);
		
		if (m_DistTable[secondIndex][m_Sorted[secondIndex][numNN]] > reachDist)
			reachDist = m_DistTable[secondIndex][m_Sorted[secondIndex][numNN]];
		
		return reachDist;		
	}
	
	private int getNN(int kNN, int instIndex) {
 
		int numNN = kNN + 1;
 
		for (int i = kNN; i < m_Instances.numInstances() - 1; i++) {
			if (m_DistTable[instIndex][m_Sorted[instIndex][i]] == m_DistTable[instIndex][m_Sorted[instIndex][i+1]])
				numNN++;
			else
				break;
		}
 
		return numNN;
	}
	
	private double LRD(int kNN, int instIndex) {
		
		// get the number of nearest neighbors:
		int numNN = getNN(kNN, instIndex);
 
		double lrd = 0;
 
		for (int i = 1; i < numNN; i++) {
			lrd += getReachDist(kNN, instIndex, m_Sorted[instIndex][i]);
		}
		lrd = numNN / lrd;
 
		return lrd;
	}	
	    
 
	// constructor
	public LOF(int kNN, String path, String trainFileName, String testFileName, String outFileName) throws Exception {
 
		try {
			FileReader reader;
			FileWriter writer;
 
			// read the training instances:
			reader = new FileReader(path + trainFileName);
			Instances trainInst = new Instances(reader);
 
			// read the test instances:
			reader = new FileReader(path + testFileName);
			Instances testInst = new Instances(reader);
 
			// output initial information:
			writer = new FileWriter(outFileName); 
			writer.append("% Training data file: " + trainFileName + "\n");
			writer.append("% Testing data file: " + testFileName + "\n");
			writer.append("%\n");
			writer.append("% Results\n");
			writer.append("%\n");
			writer.flush();
 
			// get the bounds for numeric attributes of training instances:
			getMinMax(trainInst);
 
			int numTrain = trainInst.numInstances();
			int numTest = testInst.numInstances();
			int numCurr = numTrain + 1;
			
			// prepare the current instances:
			m_Instances = new Instances(trainInst);			
			
			m_DistTable = new double[numCurr][numCurr];
			m_Sorted = new int[numCurr][numCurr];
			double [] lof = new double[numTest];
			
			// fill the table with distances among training instances:
			for (int i = 0; i < numTrain; i++) {
				m_DistTable[i][i] = 0;
				for (int j = i + 1; j < numTrain; j++) {
					m_DistTable[i][j] = getDistance(trainInst.instance(i), trainInst.instance(j));
					m_DistTable[j][i] = m_DistTable[i][j];
				}	
		    }
			
			m_DistTable[numTrain][numTrain] = 0;		
			
			for (int k = 0; k < numTest; k++) {	
				
				// update instances with the current test instance:
				if (m_Instances.numInstances() > numTrain)
					m_Instances.delete(numTrain);
				m_Instances.add(testInst.instance(k));
								
				// update the table with distances among training instances and the current test instance:
				for (int i = 0; i < numTrain; i++) {
				    m_DistTable[i][numTrain] = getDistance(m_Instances.instance(i), m_Instances.instance(numTrain));
				    m_DistTable[numTrain][i] = m_DistTable[i][numTrain];
			    }
				
				// sort the distances
				for (int i = 0; i < numCurr; i++) {
					m_Sorted[i] = Utils.sort(m_DistTable[i]);
			    }				
				
				// get the number of nearest neighbors for the current test instance:
				int numNN = getNN(kNN, numTrain);
 
				// get LOF for the current test instance:
				lof[k] = 0;
				for (int i = 1; i < numNN; i++) {
					lof[k] += LRD(kNN, m_Sorted[numTrain][i]) / LRD(kNN, numTrain);
			    }
				lof[k] = lof[k] / numNN;
				
				writer.append(lof[k] + "\n");
				writer.flush();
			}
								
			writer.append("\n\n");
 
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
	}
 
	// unit test
	public static void main(String[] args) {
 
		try  {
			
			String path = "D:/CIVaBiS/LOF/data/";
 
			//new LOF(10, path,  "events.1285.train.arff", "events.1285.test.arff", "lof.1285.events.txt");
			//new LOF(10, path,  "events.1287.train.arff", "events.1287.test.arff", "lof.1287.events.txt");
			
			new LOF(10, path,  "dataset1.arff", "dataset1.arff", "lof1.txt");
			new LOF(10, path,  "dataset2.arff", "dataset1.arff", "lof2.txt");
 
		} catch(Exception e) {
		}
	}
}
