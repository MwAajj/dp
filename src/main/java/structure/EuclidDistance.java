package structure;

import weka.core.Instance;

public final class EuclidDistance {

    public static double euclidDistance(Instance old, Instance newInstance) {
        if (old.numAttributes() != newInstance.numAttributes()) {
            throw new RuntimeException("CalculateDistance: Incompatible size of instances");
        }
        double sum = 0;
        for (int i = 0; i < old.numAttributes(); i++) {
            double value = old.value(i);
            double value1 = newInstance.value(i);
            double x = value - value1;
            if (Double.isNaN(x))
                continue;
            sum += Math.pow(x, 2);
        }
        return Math.sqrt(sum);
    }
}
