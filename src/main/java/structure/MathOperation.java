package structure;

import weka.core.Instance;
import weka.core.Instances;

public final class MathOperation {

    public static double euclidDistance(int classIndex, Instance old, Instance newInstance) {
        if (old.numAttributes() != newInstance.numAttributes()) {
            throw new RuntimeException("CalculateDistance: Incompatible size of instances");
        }
        double sum = 0;
        for (int i = 0; i < old.numAttributes(); i++) {
            double value = old.value(i);
            double value1 = newInstance.value(i);
            double x = value - value1;
            if (i == classIndex) continue; // don't calculate distance for class index
            sum += Math.pow(x, 2);
        }
        return Math.sqrt(sum);
    }

    public static double euclidDistance(int classIndex, Instance old, double[] distances, int r) {
        double sum = 0;
        for (int i = 0; i < distances.length; i++) {
            if (i == classIndex) continue; // don't calculate distance for class index
            sum += Math.pow((old.value(i) - distances[i]), 2d);
        }
        return Math.sqrt(sum);
    }

    //denominator -- down | numerator -- up
    public static double fuzzyDistance(Instances instances, Instance instance, double classValue, int m) {
        double numeratorSum = 0d, denominatorSum = 0d, upperFraction;
        int numerator;
        for (Instance value : instances) {
            numerator = value.classValue() == classValue ? 1 : 0;
            double distance = euclidDistance(value.classIndex(), value, instance);
            double pow = Math.pow(distance, m);
            upperFraction = numerator / pow;
            numeratorSum += upperFraction;
            denominatorSum += (1 / pow);
        }
        return numeratorSum / denominatorSum;
    }

    //denominator -- down | numerator -- up
    public static double hmDistance(Instances instances, Instance instance, double classValue, int k) {
        double denominatorSum = 0d;
        for (Instance value : instances) {
            if(value.classValue() != classValue) continue;
            double distance = euclidDistance(value.classIndex(), value, instance);
            double pow = Math.pow(distance, 2);
            denominatorSum += (1 / pow);
        }
        return k / denominatorSum;
    }

    public static double meanDistances(Instances instances, Instance instance, double classValue, int i) {
        double sum = 0d, result;
        for (Instance value : instances) {
            if(value.classValue() != classValue) continue;
            double distance = euclidDistance(value.classIndex(), value, instance);
            sum += distance;
        }
        result = (1 / (double) i) * sum;
        return result;
    }

    public static double harmonicDistance(Instances instances, double[] distances, double classValue, int r) {
        double result, denominatorSum = 0d;
        for (Instance value : instances) {
            if(value.classValue() != classValue) continue;
            double distance = euclidDistance(value.classIndex(), value, distances, r);
            denominatorSum += (1 / distance);
        }
        result = r / denominatorSum;
        return result;
    }

    public static double newHarmonicDistance(double distance, int k, int hm) { //harmonic mean size
        double result, denominatorSum = 0d;
        for (int i = 0; i < hm; i++) {
            denominatorSum += 1/distance;
        }
        result = k / denominatorSum;
        return result;
    }

    public static double getMaxDistance(double[] distances) {
        double max = Double.MIN_VALUE;
        for (double distance : distances)
            if (max < distance)
                max = distance;
        return max;
    }

    public static Instance getMedianInstance(Instances data, int level) {
        Instances instances = new Instances(data);
        instances.sort(level);
        return instances.get(instances.size() / 2);
    }
}
