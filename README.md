jLOF
=====

Java implementation of Local Outlier Factor algorithm by [Markus M. Breunig](http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf). The implementation accepts a collection `double[]` arrays, where each array corresponds to an instance.  

Example 1: Train and test data
--------

The following example illustrates the simple use case of computing LOF value of an example based on the training data. First, we initialize `LOF` constructor by passing `data` variable. Next, we call `getScore(double[] example, int kNN)` with an array of doubles and kNN value.  
```
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
int kNN = 5;

System.out.println(model.getScore(new double[]{2, 0}, kNN));
System.out.println(model.getScore(new double[]{0, 0}, kNN));
System.out.println(model.getScore(new double[]{10, 4}, kNN));
```
The output should be:
```
0.9439752132155199
1.4115350452823745
7.845854226211409
```

Example 2: Get actual nearest neighbors
--------
To get a list of k-nearest neighbors, call `getNeighbors(double[] example, int kNN)` method:
```
double[] testSample = new double[]{2, 0};
System.out.println("\nNeighbors of "+Arrays.toString(testSample));
ArrayList<double[]> neighbors = model.getNeighbors(testSample, kNN);
for(double[] n : neighbors){
	System.out.println(Arrays.toString(n));
}
```
The output is:
```
Neighbors of [2.0, 0.0]
[0.0, 0.0]
[1.0, 0.0]
[2.0, 1.0]
[2.0, 0.0]
[2.0, 0.0]
[2.0, 0.0]
```

Example 3: Get LOF values for training data
--------
The get the LOF values for existing training data, simply call `getTrainingScores(int knn)` method:
```
double[] scores = model.getTrainingScores(kNN);
for(int i = 0; i < scores.length; i++){
	System.out.println(Arrays.toString(data.get(i)) + "\t" + scores[i]);
}
```
The output should be:
```
[0.0, 0.0]	1.5237323873815085
[0.0, 1.0]	1.196228163810988
[1.0, 0.0]	1.071470776748656
[1.0, 1.0]	1.2335968200610736
[1.0, 2.0]	1.196228163810988
[2.0, 1.0]	1.071470776748656
[2.0, 2.0]	1.5237323873815085
[2.0, 0.0]	0.9237025720677815
[2.0, 0.0]	0.9237025720677815
[2.0, 0.0]	0.9237025720677815
[2.0, 0.0]	0.9237025720677815
```