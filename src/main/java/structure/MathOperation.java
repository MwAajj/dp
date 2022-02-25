package structure;

import weka.core.Instance;
import weka.core.Instances;

public final class MathOperation {

    public static double euclidDistance(Instance old, Instance newInstance) {
        if (old.numAttributes() != newInstance.numAttributes()) {
            throw new RuntimeException("CalculateDistance: Incompatible size of instances");
        }
        double sum = 0;
        for (int i = 0; i < old.numAttributes(); i++) {
            sum += Math.pow((old.value(i) - newInstance.value(i)), 2);
        }
        return Math.sqrt(sum);
    }


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

    //denominator -- down | numerator -- up
    //--------------------------------------------F-KNN--------------------------------------------
    public static double fuzzyDistance(Instances instances, double[] distances, double classValue, int m) {
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
    //----------------------------------------------------------------------------------------

    //--------------------------------------------HMD-KNN--------------------------------------------
    public static double meanDistance(Instances instances, Instance instance, double classValue, int i) { //2
        double sum = 0d, result;
        for (Instance value : instances) {
            if (value.classValue() != classValue) continue;
            double distance = euclidDistance(value.classIndex(), value, instance);
            sum += distance;
        }
        result = (1 / (double) i) * sum;
        return result;
    }

    public static double harmonicDistance(Instances instances, double[] distances, double classValue, int r) {  //3
        double result, denominatorSum = 0d;
        for (int i = 0; i < r; i++) {
            Instance instance = instances.instance(i);
            if (instance.classValue() != classValue) continue;
            double distance = euclidDistance(instance.classIndex(), instance, distances, r);
            denominatorSum += (1 / distance);
        }
        result = r / denominatorSum;
        return result;
    }

    public static double[] meanDistances(Instances instances, Instance instance, int k, int r, int m_NumClasses) {
        double[] meanDistances = new double[k];
        int index = 0;
        for (int i = 0; i < k; i++) {
            double prob = MathOperation.meanDistance(instances, instance, i, r);
            meanDistances[index] = prob;
            index++;
        }
        return meanDistances;
    }

    public static double[] harmonicMeanDistances(Instances instances, Instance instance, int k, int r, int m_NumClasses) {
        double[] meanDistances = MathOperation.meanDistances(instances, instance, k, r, m_NumClasses);
        int index;
        double[] harmonicMeanDistances = new double[k];
        index = 0;
        for (int j = 0; j < k; j++) {
            double prob = MathOperation.harmonicDistance(instances, meanDistances, j, r);
            harmonicMeanDistances[index] = prob;
            index++;
        }
        return harmonicMeanDistances;
    }

    public static double newHarmonicDistance(Instances instances, Instance instance, int k, int r, int m_NumClasses) { //harmonic mean size
        double[] harmonicMeanDistances = MathOperation.harmonicMeanDistances(instances, instance, k, r, m_NumClasses);
        double result, denominatorSum = 0d;
        for (int i = 0; i < m_NumClasses; i++) {
            denominatorSum += 1 / harmonicMeanDistances[i];
        }
        result = k / denominatorSum;
        return result;
    }
    //------------------------------------------------------------------------------------------------------------------
}
