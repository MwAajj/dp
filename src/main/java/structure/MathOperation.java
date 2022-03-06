package structure;

import weka.core.Instance;

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
}
