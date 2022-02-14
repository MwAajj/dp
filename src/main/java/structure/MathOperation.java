package structure;

import weka.core.Instance;

public final class MathOperation {

    public static double euclidDistance(Instance first, Instance second) {
        if (first.numAttributes() != second.numAttributes()) {
            throw new RuntimeException("CalculateDistance: Incompatible size of instances");
        }
        double sum = 0;
        for (int i = 0; i < first.numAttributes(); i++) {
            sum += Math.pow((first.value(i) - second.value(i)), 2d);
        }
        double square = Math.sqrt(sum);
        return square;
    }
}
