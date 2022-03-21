package classifier.variants.advanced;

import classifier.variants.Variant;
import classifier.variants.basic.Knn;
import lombok.Getter;
import lombok.Setter;
import classifier.structure.Structure;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

@Getter
@Setter
public class FuzzyKnn implements Variant {
    Map<Double, Double> info = new HashMap<>();
    private Structure structure;
    private int k;
    private int m;

    public FuzzyKnn(Structure structure, int k, int m) {
        this.structure = structure;
        this.k = k;
        this.m = m;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int mNumberClasses) {
        double[] result;
        Instances kNearestNeighbours = structure.findKNearestNeighbours(instance, k);
        double[] distances = structure.getDistances();
        sortInstances(kNearestNeighbours, distances);
        result = fuzzyDistance2(kNearestNeighbours, distances, m, mNumberClasses);
        return result;
    }

    //denominator -- down | numerator -- up
    private double[] fuzzyDistance2(Instances kNearestNeighbours, double[] distances, int m, int mNumberClasses) {
        double upperFraction;
        int numerator;
        double[] prob = new double[mNumberClasses];

        for (int i = 0; i < mNumberClasses; i++) {
            double numeratorSum = 0d;
            double denominatorSum = 0d;
            for (int j = 0; j < kNearestNeighbours.size(); j++) {
                numerator = (int) kNearestNeighbours.get(j).classValue() == i ? 1 : 0;
                double pow = Math.pow(distances[j], 2d / (m - 1));
                upperFraction = numerator / (pow + 0.0001d);
                numeratorSum += upperFraction;
                denominatorSum += (1 / (pow + 0.0001d));
            }
            prob[i] = numeratorSum / (denominatorSum + 0.0001d);
        }
        return prob;
    }

    public void sortInstances(Instances neighbours, double[] distances) {
        Knn.sortNearestInstances(neighbours, distances);
    }

    @Override
    public double classifyInstance(Instance instance, int mNumberClasses) {
        Instances instances = structure.findKNearestNeighbours(instance, k);
        double[] distances = structure.getDistances();
        double endClass = 0d;
        double max = -1d;

        for (int i = 0; i < mNumberClasses; i++) {
            double prob = fuzzyDistance(instances, distances, i, m);
            if (max < prob) {
                max = prob;
                endClass = i;
            }
            info.put((double) i, prob);
        }
        return endClass;
    }

    @Override
    public String getOption() {
        return "-F";
    }


    private double fuzzyDistance(Instances instances, double[] distances, double classValue, int m) {
        double numeratorSum = 0d;
        double denominatorSum = 0d;
        double upperFraction;
        int numerator;
        int index = 0;
        for (Instance value : instances) {
            numerator = value.classValue() == classValue ? 1 : 0;
            double pow = Math.pow(distances[index], m);
            upperFraction = numerator / pow;
            numeratorSum += upperFraction;
            denominatorSum += (1 / pow);
            index++;
        }
        return numeratorSum / (denominatorSum + 0.00001d);
    }
}
