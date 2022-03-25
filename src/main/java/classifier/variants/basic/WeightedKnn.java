package classifier.variants.basic;

import classifier.variants.Variant;
import classifier.structure.Structure;
import weka.core.Instance;
import weka.core.Instances;

public class WeightedKnn implements Variant {
    private final Structure structure;
    private final int k;

    public WeightedKnn(Structure structure, int k) {
        this.structure = structure;
        this.k = k;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int mNumberClasses) {
        Instances kNearestNeighbours = structure.findKNearestNeighbours(instance, k);
        double[] result = new double[mNumberClasses];
        double[] distances = structure.getDistances();
        double[] weight = new double[kNearestNeighbours.size()];
        sortInstances(kNearestNeighbours, distances);
        double total = 0d;
        for (int i = 0; i < kNearestNeighbours.numInstances(); i++) {
            Instance current = kNearestNeighbours.instance(i);
            weight[i] = 1 / (distances[i] + 0.001d); //avoid to div by zero
            result[(int) current.classValue()] += weight[i];
            total += weight[i];
        }
        for (int i = 0; i < result.length; i++) {
            result[i] = result[i] / (total + 0.001d);
        }
        return result;
    }

    public void sortInstances(Instances neighbours, double[] distances) {
        Knn.sortNearestInstances(neighbours, distances);
    }

    @Override
    public double classifyInstance(Instance instance, int mNumberClasses) {
        Instances instances = structure.findKNearestNeighbours(instance, k);
        double[] distances = structure.getDistances();
        double[] results = new double[mNumberClasses];
        double[] weights = new double[instances.size()];
        for (int i = 0; i < instances.size(); i++) {
            weights[i] = 1 / (distances[i] + 0.001d); //avoid to div by zero
            results[(int) instances.get(i).classValue()] += weights[i];
        }
        double max = Integer.MIN_VALUE;
        double endClass = -1d;

        for (int i = 0; i < results.length; i++) {
            if (max < results[i]) {
                max = results[i];
                endClass = i;
            }
        }
        return endClass;
    }

    @Override
    public String getOption() {
        return "-W";
    }

    @Override
    public void setOption(String option) {
        // No option for this variant is implemented
    }
}
