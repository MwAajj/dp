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
    public double[] distributionForInstance(Instance instance, int m_NumClasses) {
        double[] result;
        Instances kNearestNeighbours = structure.findKNearestNeighbours(instance, k);
        double[] distances = structure.getDistances();
        sortInstances(kNearestNeighbours, distances);
        result = fuzzyDistance2(kNearestNeighbours, distances, m, m_NumClasses);
        return result;
    }

    //denominator -- down | numerator -- up
    private double[] fuzzyDistance2(Instances kNearestNeighbours, double[] distances, int m, int m_NumClasses) {
        double upperFraction;
        int numerator;
        double[] prob = new double[m_NumClasses];

        for (int i = 0; i < m_NumClasses; i++) {
            double numeratorSum = 0d, denominatorSum = 0d;
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
    public double classifyInstance(Instance instance, int m_NumClasses) {
        Instances instances = structure.findKNearestNeighbours(instance, k);
        double[] distances = structure.getDistances();
        double endClass = 0d, max = -1d;

        for (int i = 0; i < m_NumClasses; i++) {
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
        double numeratorSum = 0d, denominatorSum = 0d, upperFraction;
        int numerator, index = 0;
        for (Instance value : instances) {
            numerator = value.classValue() == classValue ? 1 : 0;
            double pow = Math.pow(distances[index], m);
            upperFraction = numerator / pow;
            numeratorSum += upperFraction;
            denominatorSum += (1 / pow);
            index++;
        }
        return numeratorSum / denominatorSum;
    }

    private void printInfo() {
        System.out.println("\n--------------------F-KNN------------");
        System.out.println(info);
        System.out.println("-------------------------------------");
    }
}
