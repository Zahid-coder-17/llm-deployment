package com.embeddra.edge

import android.content.Context
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.HashMap
import java.util.regex.Pattern

class VocabularyDecoder(private val context: Context) {

    private val vocabList = ArrayList<String>(128000)
    private val rawTokenToIdMap = HashMap<String, Int>(128000)
    private val byteToTokenId = IntArray(256)
    // key: (idA << 32) | idB, value: (rank << 32) | mergedId
    private val mergeMap = HashMap<Long, Long>(300000)
    private var isLoaded = false

    private val preTokenPattern = Pattern.compile(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    )

    private fun buildByteToUnicode(): CharArray {
        val bs = ArrayList<Int>(256)
        val cs = ArrayList<Int>(256)
        for (b in '!'.code..'~'.code) { bs.add(b); cs.add(b) }
        for (b in 0xA1..0xAC) { bs.add(b); cs.add(b) }
        for (b in 0xAE..0xFF) { bs.add(b); cs.add(b) }
        var n = 0
        for (b in 0..255) {
            if (!bs.contains(b)) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        val map = CharArray(256)
        for (idx in bs.indices) {
            map[bs[idx]] = cs[idx].toChar()
        }
        return map
    }

    fun loadVocabulary(): Boolean {
        if (isLoaded) return true
        try {
            val inputStream: InputStream = context.assets.open("vocab.bin")
            val bytes = inputStream.readBytes()
            inputStream.close()

            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val numTokens = buffer.int

            for (i in 0 until numTokens) {
                val len = buffer.short.toInt() and 0xFFFF
                val strBytes = ByteArray(len)
                buffer.get(strBytes)
                val rawStr = String(strBytes, Charsets.UTF_8)
                val displayStr = rawStr.replace("Ġ", " ").replace("Ċ", "\n")
                vocabList.add(displayStr)
                rawTokenToIdMap[rawStr] = i
            }

            val b2u = buildByteToUnicode()
            for (b in 0..255) {
                byteToTokenId[b] = rawTokenToIdMap[b2u[b].toString()] ?: 0
            }

            val mStream: InputStream = context.assets.open("merges.bin")
            val mBytes = mStream.readBytes()
            mStream.close()
            val mBuf = ByteBuffer.wrap(mBytes).order(ByteOrder.LITTLE_ENDIAN)
            val numMerges = mBuf.int
            for (rank in 0 until numMerges) {
                val idA = mBuf.int
                val idB = mBuf.int
                val mergedId = mBuf.int
                val key = (idA.toLong() shl 32) or (idB.toLong() and 0xFFFFFFFFL)
                val value = (rank.toLong() shl 32) or (mergedId.toLong() and 0xFFFFFFFFL)
                mergeMap[key] = value
            }

            isLoaded = true
            return true
        } catch (e: Exception) {
            e.printStackTrace()
            return false
        }
    }

    fun decodeToken(tokenId: Int): String {
        if (!isLoaded || tokenId < 0 || tokenId >= vocabList.size) {
            return ""
        }
        return vocabList[tokenId]
    }

    fun encode(rawText: String): IntArray {
        if (!isLoaded || rawText.isEmpty()) return intArrayOf(128000)

        val result = ArrayList<Int>(rawText.length)
        result.add(128000) // <|begin_of_text|>

        val matcher = preTokenPattern.matcher(rawText)
        while (matcher.find()) {
            val chunk = matcher.group() ?: continue
            val chunkBytes = chunk.toByteArray(Charsets.UTF_8)
            val word = ArrayList<Int>(chunkBytes.size)
            for (b in chunkBytes) {
                word.add(byteToTokenId[b.toInt() and 0xFF])
            }

            while (word.size > 1) {
                var bestRank = Long.MAX_VALUE
                var bestIdx = -1
                var bestMerged = -1

                for (i in 0 until word.size - 1) {
                    val key = (word[i].toLong() shl 32) or (word[i + 1].toLong() and 0xFFFFFFFFL)
                    val entry = mergeMap[key] ?: continue
                    val rank = entry ushr 32
                    if (rank < bestRank) {
                        bestRank = rank
                        bestIdx = i
                        bestMerged = (entry and 0xFFFFFFFFL).toInt()
                    }
                }

                if (bestIdx == -1) break
                word[bestIdx] = bestMerged
                word.removeAt(bestIdx + 1)
            }

            result.addAll(word)
        }

        return result.toIntArray()
    }
}
