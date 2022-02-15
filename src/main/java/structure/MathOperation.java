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
            if(i == classIndex) continue; // don't calculate distance for class index
            sum += Math.pow((old.value(i) - newInstance.value(i)), 2d);
        }
        return Math.sqrt(sum);
    }

    //denominator -- down | numerator -- up
    public static double fuzzyDistance(Instances instances,  Instance instance, double classValue,  int m) {
        double numeratorSum = 0d, denominatorSum = 0d,  upperFraction = 0d;
        int numerator;
        for (int i = 0; i < instances.size(); i++) {
            numerator = instances.get(i).classValue() == classValue ? 1 : 0;
            double distance = euclidDistance(instances.get(i).classIndex(), instances.get(i), instance);
            double pow = Math.pow(distance, m);
            upperFraction = numerator / pow;
            numeratorSum += upperFraction;
            denominatorSum += (1/ pow);
        }
        return numeratorSum / denominatorSum;
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
